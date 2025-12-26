# Python 3.11 slim 이미지 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 빌드 도구 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY app.py .

# 결과 파일 저장용 디렉토리 생성
RUN mkdir -p /tmp/heat_eq_results

# 포트 노출 (fly.io가 자동으로 매핑)
EXPOSE 8080

# 환경 변수 설정
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# 애플리케이션 실행
# fly.io는 PORT 환경 변수를 제공하므로 app.py에서 자동으로 읽음
CMD ["python", "app.py"]

