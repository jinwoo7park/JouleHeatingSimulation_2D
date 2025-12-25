"""
Vercel serverless function wrapper for Flask app
Converts Flask WSGI app to AWS Lambda handler format (Vercel Python runtime)
"""
import sys
import os
import json
import base64
from io import BytesIO

# Add parent directory to path to import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

def handler(event, context):
    """AWS Lambda handler that wraps Flask WSGI app"""
    # Parse Vercel event (supports multiple formats)
    # Try different event formats for compatibility
    http_method = 'GET'
    if 'httpMethod' in event:
        http_method = event['httpMethod']
    elif 'requestContext' in event and 'http' in event['requestContext']:
        http_method = event['requestContext']['http'].get('method', 'GET')
    elif 'method' in event:
        http_method = event['method']
    
    # Parse path
    path = '/'
    if 'path' in event:
        path = event['path']
        # Remove /api prefix if present (since we're routing to /api/index.py)
        if path.startswith('/api'):
            path = path[3:] if len(path) > 3 else '/'
    elif 'rawPath' in event:
        path = event['rawPath']
        if path.startswith('/api'):
            path = path[3:] if len(path) > 3 else '/'
    
    # Ensure path starts with /
    if not path.startswith('/'):
        path = '/' + path
    
    # Parse query string
    query_string = ''
    if 'queryStringParameters' in event and event['queryStringParameters']:
        query_params = event['queryStringParameters']
        if isinstance(query_params, dict):
            query_string = '&'.join([f"{k}={v}" for k, v in query_params.items()])
    elif 'rawQueryString' in event:
        query_string = event['rawQueryString']
    
    # Parse headers
    headers = event.get('headers', {})
    if not isinstance(headers, dict):
        headers = {}
    
    # Parse body
    body = event.get('body', '')
    if event.get('isBase64Encoded', False) and body:
        body = base64.b64decode(body).decode('utf-8')
    
    # Prepare body bytes for wsgi.input
    body_bytes = body.encode('utf-8') if body else b''
    
    # Build WSGI environ
    environ = {
        'REQUEST_METHOD': http_method,
        'PATH_INFO': path,
        'QUERY_STRING': query_string,
        'CONTENT_TYPE': headers.get('content-type', ''),
        'CONTENT_LENGTH': str(len(body_bytes)),
        'SERVER_NAME': 'localhost',
        'SERVER_PORT': '80',
        'wsgi.version': (1, 0),
        'wsgi.url_scheme': headers.get('x-forwarded-proto', 'https'),
        'wsgi.input': BytesIO(body_bytes),
        'wsgi.errors': sys.stderr,
        'wsgi.multithread': False,
        'wsgi.multiprocess': True,
        'wsgi.run_once': False,
        'HTTP_HOST': headers.get('host', ''),
    }
    
    # Add HTTP headers
    for key, value in headers.items():
        key = key.upper().replace('-', '_')
        if key not in ('CONTENT_TYPE', 'CONTENT_LENGTH'):
            environ[f'HTTP_{key}'] = value
    
    # Collect response
    response_status = 200
    response_headers = []
    response_body = []
    
    def start_response(status, headers):
        nonlocal response_status
        response_status = int(status.split(' ', 1)[0])
        response_headers.extend(headers)
    
    # Run Flask app
    try:
        for chunk in app(environ, start_response):
            response_body.append(chunk)
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Flask app error: {str(e)}")
        print(f"Traceback: {error_traceback}")
        response_status = 500
        error_response = {
            'error': str(e),
            'error_type': type(e).__name__,
            'traceback': error_traceback
        }
        response_body = [json.dumps(error_response).encode('utf-8')]
        if not response_headers:
            response_headers = [('Content-Type', 'application/json')]
    
    # Build response body
    body_bytes = b''.join(response_body)
    body_str = body_bytes.decode('utf-8') if body_bytes else ''
    
    # Convert headers to dict
    headers_dict = {}
    for key, value in response_headers:
        headers_dict[key.lower()] = value
    
    # Return Lambda response
    return {
        'statusCode': response_status,
        'headers': headers_dict,
        'body': body_str
    }

