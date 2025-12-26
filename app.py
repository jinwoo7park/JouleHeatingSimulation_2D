from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
import json
import traceback
import os
import threading
import time
import sys
import atexit
from numba import njit
from concurrent.futures import ThreadPoolExecutor
import tempfile
import shutil

app = Flask(__name__)
CORS(app)

# DoS 방지를 위한 입력 제한 설정
# 요청 바디 크기 제한: 10MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB

# 그리드 크기 제한 (DoS 방지)
MAX_NR = 120  # r 방향 최대 그리드 수
MAX_NZ = 400  # z 방향 최대 그리드 수
MAX_N_TOTAL = 50000  # 총 노드 수 최대값 (Nr * Nz)

# 시간 범위 제한
MAX_T_END = 1e6  # 최대 시뮬레이션 시간 (초, 약 11.6일)
MAX_T_EVAL_POINTS = 100  # 최대 시간 포인트 수

        # 진행률 저장용 전역 변수 (스레드 안전)
progress_store = {}
progress_lock = threading.Lock()

# ThreadPoolExecutor로 동시 실행 수 제한 (최대 3개 작업 동시 실행)
# 프로덕션 환경에서는 Celery/RQ + Redis 권장
executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="sim_worker")

# 결과 파일 저장 디렉토리
RESULTS_DIR = os.path.join(tempfile.gettempdir(), 'heat_eq_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 출력 버퍼 강제 플러시 함수
def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

# 진행률 정리 함수 (주기적으로 실행)
def cleanup_old_progress():
    """오래된 진행률 데이터 정리
    - 완료된 세션(progress>=100): 5분 후 삭제
    - 에러 세션: 30분 후 삭제
    - 결과 파일도 함께 삭제"""
    import time
    current_time = time.time()
    with progress_lock:
        to_remove = []
        for sid, progress_data in progress_store.items():
            if 'timestamp' not in progress_data:
                continue
            
            age = current_time - progress_data['timestamp']
            
            # 완료된 세션: 5분 후 삭제
            if progress_data.get('progress', 0) >= 100 and age > 300:
                to_remove.append(sid)
            # 에러 세션: 30분 후 삭제
            elif progress_data.get('error') is not None and age > 1800:
                to_remove.append(sid)
        
        for sid in to_remove:
            # progress_store에서 삭제
            if sid in progress_store:
                del progress_store[sid]
            
            # 결과 파일 삭제
            result_file = os.path.join(RESULTS_DIR, f"{sid}.npz")
            if os.path.exists(result_file):
                try:
                    os.remove(result_file)
                    flush_print(f"🗑️ 오래된 결과 파일 삭제: {sid}")
                except Exception as e:
                    flush_print(f"⚠️ 결과 파일 삭제 실패 ({sid}): {e}")

# 주기적으로 진행률 정리하는 백그라운드 스레드
cleanup_thread_running = threading.Event()
cleanup_thread_running.set()  # 시작 시 True

def periodic_cleanup_worker():
    """주기적으로 cleanup_old_progress를 실행하는 백그라운드 스레드
    서버가 조용해도 파일이 계속 정리되도록 함"""
    while cleanup_thread_running.is_set():
        try:
            time.sleep(60)  # 1분마다 실행
            cleanup_old_progress()
        except Exception as e:
            flush_print(f"⚠️ 주기적 정리 중 오류: {e}")

# 백그라운드 정리 스레드 시작
cleanup_thread = threading.Thread(target=periodic_cleanup_worker, daemon=True, name="cleanup_worker")
cleanup_thread.start()

# 프로세스 종료 시 정리 함수
def cleanup_on_exit():
    """프로세스 종료 시 실행되는 정리 함수"""
    flush_print("🛑 프로세스 종료 중... 정리 작업 수행")
    
    # 백그라운드 정리 스레드 중지
    cleanup_thread_running.clear()
    if cleanup_thread.is_alive():
        cleanup_thread.join(timeout=2.0)
    
    # ThreadPoolExecutor 종료 (대기 중인 작업은 취소)
    executor.shutdown(wait=False)
    flush_print("✅ ThreadPoolExecutor 종료됨")
    
    # 마지막으로 한 번 더 정리 실행
    cleanup_old_progress()
    flush_print("✅ 정리 작업 완료")

# atexit에 정리 함수 등록
atexit.register(cleanup_on_exit)

# Numba로 가속된 라플라시안 코어 함수 (모듈 전역으로 정의하여 재컴파일 방지)
@njit
def _build_sparse_laplacian_core(Nr, Nz, N_total, r, dr_cell, dz_cell, k_r_grid, k_z_grid, rho_cp_grid):
    """2D 원통좌표계 라플라시안 스파스 행렬 구성 (Numba 가속 코어)
    FVM 보존형 이산화로 r=0 특이점을 올바르게 처리
    
    r 방향 FVM:
    - 셀 i의 부피: V_i = 2π * r_i * Δr_i * dz (i > 0), V_0 = π * (r_{1/2})^2 * dz (i = 0)
    - 플럭스: F_{i+1/2} = 2π * r_{i+1/2} * k * (T_{i+1} - T_i) / Δr_{i+1/2}
    - 계수: F_{i+1/2} / (V_i * ρcp) = r_{i+1/2} * k / (r_i * Δr_i * Δr_{i+1/2} * ρcp)
    여기서 Δr_i는 셀 i의 두께, Δr_{i+1/2}는 인터페이스 간격
    이렇게 하면 1/Δr² 스케일이 보장됨
    """
    max_elements = N_total * 5  # 각 노드당 최대 5개 요소
    data = np.zeros(max_elements)
    rows = np.zeros(max_elements, dtype=np.int32)
    cols = np.zeros(max_elements, dtype=np.int32)
    idx_count = 0

    for i in range(Nr):
        for j in range(Nz):
            idx = i * Nz + j

            # 이방성 열전도도
            k_r_center = k_r_grid[i, j]
            k_z_center = k_z_grid[i, j]
            rho_cp = rho_cp_grid[i, j]

            # r 방향 계수 (FVM 방식)
            coeff_r_up = 0.0
            coeff_r_down = 0.0

            if i == 0:
                # r=0 (축): 대칭 조건 ∂T/∂r = 0
                # FVM: r_{-1/2} = 0이므로, r_{1/2} 방향 플럭스만 존재
                # 
                # 정확한 FVM:
                # r=0 셀의 부피: V_0 = π * (r_{1/2})^2 * dz, 여기서 r_{1/2} = r[1]/2
                # 플럭스: F_{1/2} = 2π * r_{1/2} * k * (T1 - T0) / Δr_{1/2}
                # 인터페이스 간격: Δr_{1/2} = r[1] - r[0] = dr_cell[0] (일관된 정의)
                # 계수: F_{1/2} / (V_0 * ρcp) = [2π * r_{1/2} * k / Δr_{1/2}] / [π * r_{1/2}^2 * dz * ρcp]
                #      = 2 * k / (r_{1/2} * Δr_{1/2} * ρcp)
                # 균일 격자에서: r_{1/2} = Δr/2, Δr_{1/2} = Δr
                # 따라서: coeff = 2k / ((Δr/2) * Δr * ρcp) = 4k/(Δr² ρcp) ✓
                if i < Nr - 1:
                    k_r_down = k_r_grid[i + 1, j]
                    k_r_interface = 2.0 * k_r_center * k_r_down / (k_r_center + k_r_down)
                    # 인터페이스 간격: r[0]에서 r[1]까지의 거리 (일관된 정의)
                    dr_interface = dr_cell[0]  # Δr_{1/2} = r[1] - r[0] = Δr
                    r_half = r[1] * 0.5  # r_{1/2} = r[1]/2 = Δr/2
                    # r=0에서: 계수 = 2 * k / (r_{1/2} * Δr_{1/2} * ρcp)
                    # 균일 격자에서 4k/(Δr² ρcp) 스케일로 정확히 떨어짐
                    coeff_r_down = 2.0 * k_r_interface / (r_half * dr_interface * rho_cp)

                    idx_down = (i + 1) * Nz + j
                    data[idx_count] = coeff_r_down
                    rows[idx_count] = idx
                    cols[idx_count] = idx_down
                    idx_count += 1
            else:
                # i > 0: 일반적인 경우
                # FVM 보존형 이산화 (정의 A: 셀 중심 기반):
                # 셀 중심이 r[i]에 있다고 가정:
                # 셀 i는 r[i-1/2]와 r[i+1/2] 사이에 있음
                # r[i-1/2] = (r[i-1] + r[i]) / 2, r[i+1/2] = (r[i] + r[i+1]) / 2
                # 셀 i의 두께: Δr_i = r[i+1/2] - r[i-1/2] = (r[i+1] - r[i-1]) / 2
                #              = (dr_cell[i-1] + dr_cell[i]) / 2 (i > 0이고 i < Nr-1인 경우)
                # 
                # 셀 i의 부피: V_i = 2π * r_i * Δr_i * dz
                # 플럭스: F_{i+1/2} = 2π * r_{i+1/2} * k * (T_{i+1} - T_i) / Δr_{i+1/2}
                # 계수: F_{i+1/2} / (V_i * ρcp) = r_{i+1/2} * k / (r_i * Δr_i * Δr_{i+1/2} * ρcp)
                # 여기서 Δr_{i+1/2} = r[i+1] - r[i] = dr_cell[i] (인터페이스 간격)
                # 이렇게 하면 1/Δr² 스케일이 보장됨
                
                # 셀 i의 두께를 일관되게 정의 (위쪽과 아래쪽 모두에서 동일하게 사용)
                if i > 0 and i < Nr - 1:
                    # 표준 셀 중심 기반 FVM: Δr_i = (dr_cell[i-1] + dr_cell[i]) / 2
                    dr_cell_i = (dr_cell[i - 1] + dr_cell[i]) * 0.5
                elif i == Nr - 1:
                    # 마지막 셀 (경계): r_{i+1/2}가 없으므로 r_{i-1/2}만 사용
                    # Δr_i = r[i] - r_{i-1/2} = dr_cell[i-1]
                    dr_cell_i = dr_cell[i - 1]
                else:
                    # i > 0인데 위 조건에 안 맞는 경우 (방어 코드)
                    dr_cell_i = dr_cell[i - 1] if i > 0 else dr_cell[0]
                
                # 위쪽 (i-1, j)
                if i > 0:
                    k_r_up = k_r_grid[i - 1, j]
                    k_r_interface_up = 2.0 * k_r_center * k_r_up / (k_r_center + k_r_up)
                    # 인터페이스 i-1/2의 간격: r[i-1]에서 r[i]까지의 거리
                    dr_interface_up = dr_cell[i - 1]  # Δr_{i-1/2} = r[i] - r[i-1]
                    r_interface_up = (r[i - 1] + r[i]) * 0.5  # r_{i-1/2}
                    # 계수: r_{i-1/2} * k / (r_i * Δr_i * Δr_{i-1/2} * ρcp)
                    # dr_cell_i는 위에서 일관되게 정의됨
                    coeff_r_up = k_r_interface_up * r_interface_up / (r[i] * dr_cell_i * dr_interface_up * rho_cp)

                    idx_up = (i - 1) * Nz + j
                    data[idx_count] = coeff_r_up
                    rows[idx_count] = idx
                    cols[idx_count] = idx_up
                    idx_count += 1

                # 아래쪽 (i+1, j)
                if i < Nr - 1:
                    k_r_down = k_r_grid[i + 1, j]
                    k_r_interface_down = 2.0 * k_r_center * k_r_down / (k_r_center + k_r_down)
                    # 인터페이스 i+1/2의 간격: r[i]에서 r[i+1]까지의 거리
                    dr_interface_down = dr_cell[i]  # Δr_{i+1/2} = r[i+1] - r[i]
                    r_interface_down = (r[i] + r[i + 1]) * 0.5  # r_{i+1/2}
                    # 계수: r_{i+1/2} * k / (r_i * Δr_i * Δr_{i+1/2} * ρcp)
                    # dr_cell_i는 위에서 일관되게 정의됨 (위쪽과 동일)
                    coeff_r_down = k_r_interface_down * r_interface_down / (r[i] * dr_cell_i * dr_interface_down * rho_cp)

                    idx_down = (i + 1) * Nz + j
                    data[idx_count] = coeff_r_down
                    rows[idx_count] = idx
                    cols[idx_count] = idx_down
                    idx_count += 1

            # z 방향 계수 (FVM 보존형 이산화)
            # FVM: ∂/∂z (k ∂T/∂z) → 플럭스 / (control volume 두께)
            # 인터페이스 간격: Δz_{j+1/2} = z[j+1] - z[j] = dz_cell[j]
            # Control volume 두께: Δz_j = (Δz_{j-1/2} + Δz_{j+1/2}) / 2 = (dz_cell[j-1] + dz_cell[j]) / 2
            # 플럭스: F_{j+1/2} = k * (T[j+1] - T[j]) / Δz_{j+1/2}
            # 계수: F_{j+1/2} / (Δz_j * ρcp) = k / (Δz_j * Δz_{j+1/2} * ρcp)
            # 이렇게 하면 1/Δz² 스케일이 보장됨
            coeff_z_left = 0.0
            coeff_z_right = 0.0

            # 왼쪽 (i, j-1)
            if j > 0:
                k_z_left = k_z_grid[i, j - 1]
                k_z_interface = 2.0 * k_z_center * k_z_left / (k_z_center + k_z_left)
                # 인터페이스 j-1/2의 간격: z[j-1]에서 z[j]까지의 거리
                dz_interface_left = dz_cell[j - 1]  # Δz_{j-1/2} = z[j] - z[j-1]
                # Control volume j의 두께: Δz_j = (Δz_{j-1/2} + Δz_{j+1/2}) / 2
                if j < len(dz_cell):
                    dz_control_volume = (dz_cell[j - 1] + dz_cell[j]) * 0.5
                else:
                    dz_control_volume = dz_cell[j - 1]  # 마지막 셀의 경우
                # 계수: k / (Δz_j * Δz_{j-1/2} * ρcp)
                # 이렇게 하면 1/Δz² 스케일이 보장됨
                coeff_z_left = k_z_interface / (dz_control_volume * dz_interface_left * rho_cp)

                idx_left = i * Nz + (j - 1)
                data[idx_count] = coeff_z_left
                rows[idx_count] = idx
                cols[idx_count] = idx_left
                idx_count += 1

            # 오른쪽 (i, j+1)
            if j < Nz - 1:
                k_z_right = k_z_grid[i, j + 1]
                k_z_interface = 2.0 * k_z_center * k_z_right / (k_z_center + k_z_right)
                # 인터페이스 j+1/2의 간격: z[j]에서 z[j+1]까지의 거리
                dz_interface_right = dz_cell[j]  # Δz_{j+1/2} = z[j+1] - z[j]
                # Control volume j의 두께: 위와 동일하게 (dz_cell[j-1] + dz_cell[j]) / 2
                if j > 0 and j < len(dz_cell):
                    dz_control_volume = (dz_cell[j - 1] + dz_cell[j]) * 0.5
                elif j > 0:
                    dz_control_volume = dz_cell[j - 1]
                elif j < len(dz_cell):
                    dz_control_volume = dz_cell[j]
                else:
                    dz_control_volume = dz_cell[0] if len(dz_cell) > 0 else 1e-9
                # 계수: k / (Δz_j * Δz_{j+1/2} * ρcp)
                # 이렇게 하면 1/Δz² 스케일이 보장됨
                coeff_z_right = k_z_interface / (dz_control_volume * dz_interface_right * rho_cp)

                idx_right = i * Nz + (j + 1)
                data[idx_count] = coeff_z_right
                rows[idx_count] = idx
                cols[idx_count] = idx_right
                idx_count += 1

            # 중심점 계수 (이웃 계수의 음수 합)
            center_coeff = 0.0

            if i == 0:
                # r=0: 아래쪽 계수만 있음 (이미 2배가 포함됨)
                if i < Nr - 1:
                    center_coeff -= coeff_r_down
            else:
                # 일반적인 경우: 위쪽과 아래쪽 계수
                if i > 0:
                    center_coeff -= coeff_r_up
                if i < Nr - 1:
                    center_coeff -= coeff_r_down

            if j > 0:
                center_coeff -= coeff_z_left
            if j < Nz - 1:
                center_coeff -= coeff_z_right

            data[idx_count] = center_coeff
            rows[idx_count] = idx
            cols[idx_count] = idx
            idx_count += 1

    return data, rows, cols, idx_count

@app.errorhandler(RequestEntityTooLarge)
def handle_request_entity_too_large(e):
    """요청 바디 크기 제한 초과 시 에러 처리"""
    return jsonify({
        'success': False,
        'error': f'요청 크기가 너무 큽니다. 최대 크기: {app.config["MAX_CONTENT_LENGTH"] / (1024*1024):.1f}MB'
    }), 413

@app.route('/')
def health():
    return jsonify({'status': 'ok', 'message': 'Flask backend is running'})

@app.route('/api/progress/<session_id>', methods=['GET'])
def get_progress(session_id):
    """시뮬레이션 진행률 조회
    단순화: progress_store만 사용 (캐시 제거)"""
    try:
        with progress_lock:
            progress = progress_store.get(session_id, {'progress': 0, 'message': '시작 전'})
        
        # 결과가 있으면 반환
        response = {
            'progress': progress.get('progress', 0),
            'message': progress.get('message', '시작 전'),
            'has_result': 'result' in progress or 'result_path' in progress,
            'has_error': 'error' in progress
        }
        
        # 에러 정보 포함
        if 'error' in progress:
            response['error'] = progress.get('error')
        
        # 결과 데이터가 있으면 포함 (프론트엔드에서 바로 사용 가능)
        if 'result' in progress:
            response['result'] = progress.get('result')
        
        # 메타데이터도 포함 (호환성 유지)
        if 'result_metadata' in progress:
            response['result_metadata'] = progress.get('result_metadata')
        
        return jsonify(response)
    except Exception as e:
        flush_print(f"⚠️ 진행률 조회 에러 (session_id={session_id}): {e}")
        flush_print(traceback.format_exc())
        return jsonify({
            'progress': 0,
            'message': f'진행률 조회 중 오류 발생: {str(e)}',
            'has_result': False,
            'has_error': True,
            'error': str(e)
        }), 500

@app.route('/api/result/<session_id>', methods=['GET'])
def get_result(session_id):
    """결과 파일 다운로드 (npz 형식)
    전체 데이터가 필요한 경우 사용"""
    with progress_lock:
        progress = progress_store.get(session_id)
        if not progress:
            return jsonify({'error': '세션을 찾을 수 없습니다.'}), 404
        
        result_path = progress.get('result_path')
        if not result_path or not os.path.exists(result_path):
            # 레거시: 메모리에 결과가 있으면 반환
            if 'result' in progress:
                return jsonify(progress['result'])
            return jsonify({'error': '결과 파일을 찾을 수 없습니다.'}), 404
    
    # npz 파일 스트리밍
    return send_file(
        result_path,
        mimetype='application/octet-stream',
        as_attachment=True,
        download_name=f'result_{session_id}.npz'
    )

@app.route('/api/debug/progress-store', methods=['GET'])
def debug_progress_store():
    """디버깅용: progress_store 전체 내용 조회"""
    with progress_lock:
        # 세션 ID 목록과 간단한 정보만 반환 (메모리 절약)
        store_summary = {}
        for sid, data in progress_store.items():
            store_summary[sid] = {
                'progress': data.get('progress', 0),
                'message': data.get('message', 'N/A'),
                'timestamp': data.get('timestamp', 0),
                'has_result': 'result_path' in data or 'result' in data,
                'has_error': 'error' in data
            }
        
        return jsonify({
            'progress_store': store_summary,
            'total_sessions': len(progress_store)
        })

@app.route('/api/simulate', methods=['POST'])
def simulate():
    """시뮬레이션 요청을 받아 즉시 session_id를 반환하고, 실제 시뮬레이션은 별도 스레드에서 실행"""
    import uuid
    import sys
    session_id = str(uuid.uuid4())
    
    # 진행률 초기화 (lock으로 안전하게)
    with progress_lock:
        progress_store[session_id] = {'progress': 0, 'message': '초기화 중...', 'timestamp': time.time()}
    
    # 요청 데이터 저장 (스레드에서 사용)
    data = request.json
    if not data:
        return jsonify({'success': False, 'error': '요청 데이터가 없습니다.'}), 400
    
    # ThreadPoolExecutor로 시뮬레이션 실행 (동시 실행 수 제한)
    def run_simulation():
        try:
            _simulate_worker(session_id, data)
        except Exception as e:
            error_msg = f"시뮬레이션 실행 중 오류: {str(e)}"
            flush_print(f"❌ {error_msg}")
            flush_print(f"트레이스백:\n{traceback.format_exc()}")
            with progress_lock:
                progress_store[session_id] = {
                    'progress': 0, 
                    'message': f'오류 발생: {str(e)}',
                    'error': str(e),
                    'timestamp': time.time()
                }
    
    # ThreadPoolExecutor로 제출 (daemon thread 대신)
    executor.submit(run_simulation)
    
    # 즉시 session_id 반환
    return jsonify({
        'success': True,
        'session_id': session_id,
        'message': '시뮬레이션이 시작되었습니다. /api/progress/<session_id>로 진행률을 확인하세요.'
    })

def _simulate_worker(session_id, data):
    """실제 시뮬레이션 작업을 수행하는 워커 함수"""
    try:
        # 오래된 진행률 정리
        cleanup_old_progress()
        
        flush_print(f"=== 시뮬레이션 요청 받음 (session_id: {session_id}) ===")
        
        if not data:
            raise ValueError("요청 데이터가 없습니다.")
        
        flush_print(f"요청 데이터 키: {list(data.keys())}")
        
        # 파라미터 추출 및 입력 검증
        layer_names = data.get('layer_names', [])
        if not layer_names:
            raise ValueError("layer_names가 없습니다.")
        
        k_therm_layers_original = np.array(data.get('k_therm_layers', []))
        rho_layers = np.array(data.get('rho_layers', []))
        c_p_layers = np.array(data.get('c_p_layers', []))
        thickness_layers_nm_original = np.array(data.get('thickness_layers_nm', []))
        
        # 배열 길이 일치 검사
        n_layers = len(layer_names)
        if len(k_therm_layers_original) != n_layers:
            raise ValueError(f"k_therm_layers 길이({len(k_therm_layers_original)})가 layer_names 길이({n_layers})와 일치하지 않습니다.")
        if len(rho_layers) != n_layers:
            raise ValueError(f"rho_layers 길이({len(rho_layers)})가 layer_names 길이({n_layers})와 일치하지 않습니다.")
        if len(c_p_layers) != n_layers:
            raise ValueError(f"c_p_layers 길이({len(c_p_layers)})가 layer_names 길이({n_layers})와 일치하지 않습니다.")
        if len(thickness_layers_nm_original) != n_layers:
            raise ValueError(f"thickness_layers_nm 길이({len(thickness_layers_nm_original)})가 layer_names 길이({n_layers})와 일치하지 않습니다.")
        
        # 두께 검증 (음수/0 체크)
        if np.any(thickness_layers_nm_original <= 0):
            invalid_indices = np.where(thickness_layers_nm_original <= 0)[0]
            invalid_layers = [layer_names[i] for i in invalid_indices]
            raise ValueError(f"두께가 0 이하인 레이어가 있습니다: {invalid_layers}")
        
        # 물성값 검증 (음수 체크)
        if np.any(k_therm_layers_original < 0):
            raise ValueError("k_therm_layers에 음수 값이 있습니다.")
        if np.any(rho_layers < 0):
            raise ValueError("rho_layers에 음수 값이 있습니다.")
        if np.any(c_p_layers < 0):
            raise ValueError("c_p_layers에 음수 값이 있습니다.")
        
        flush_print(f"레이어 수: {len(layer_names)}")
        
        # 소자 크기 (mm²) 입력 및 반지름 계산
        device_area_mm2 = data.get('device_area_mm2', 1.0)  # 기본값 1 mm²
        device_radius_m = np.sqrt(device_area_mm2 / np.pi) * 1e-3  # mm²를 m²로 변환 후 반지름 계산
        
        # 물리 모델 일관성 유지: 두께 압축 제거, 원래 두께와 물성 사용
        # 계산량 감소는 그리드 포인트 수 조절로 달성 (두꺼운 층은 적은 포인트, 얇은 층은 많은 포인트)
        thickness_layers_nm = thickness_layers_nm_original.copy()
        thickness_layers = thickness_layers_nm * 1e-9
        
        # 물성을 원래 값으로 사용 (압축 보정 없음)
        k_therm_layers = k_therm_layers_original.copy()
        rho_layers_effective = rho_layers.copy()
        c_p_layers_effective = c_p_layers.copy()
        
        # 입력 파라미터 검증
        voltage = data.get('voltage')
        current_density_mA_per_cm2 = data.get('current_density')  # 입력 단위: mA/cm²
        eqe = data.get('eqe', 0.2)
        
        # 입력 범위 검증
        if voltage is None or voltage < 0:
            raise ValueError("voltage가 없거나 음수입니다.")
        if current_density_mA_per_cm2 is None or current_density_mA_per_cm2 < 0:
            raise ValueError("current_density가 없거나 음수입니다.")
        if not (0 <= eqe <= 1):
            raise ValueError(f"eqe는 0~1 사이여야 합니다. 현재 값: {eqe}")
        
        # current_density 단위 변환: mA/cm² → A/m²
        # 1 mA/cm² = 0.001 A / (0.01 m)² = 0.001 A / 0.0001 m² = 10 A/m²
        current_density_A_per_m2 = current_density_mA_per_cm2 * 10.0
        
        Q_A = voltage * current_density_A_per_m2 * (1 - eqe)  # W/m²
        
        epsilon_top = data.get('epsilon_top')
        epsilon_bottom = data.get('epsilon_bottom')
        epsilon_side = data.get('epsilon_side', 0.05)  # 측면 방사율 (기본값 0.05)
        
        # 방사율 범위 검증 (0~1)
        if epsilon_top is None or not (0 <= epsilon_top <= 1):
            raise ValueError(f"epsilon_top는 0~1 사이여야 합니다. 현재 값: {epsilon_top}")
        if epsilon_bottom is None or not (0 <= epsilon_bottom <= 1):
            raise ValueError(f"epsilon_bottom는 0~1 사이여야 합니다. 현재 값: {epsilon_bottom}")
        if not (0 <= epsilon_side <= 1):
            raise ValueError(f"epsilon_side는 0~1 사이여야 합니다. 현재 값: {epsilon_side}")
        
        sigma = 5.67e-8
        h_conv = data.get('h_conv')
        T_ambient = data.get('T_ambient')
        
        # 대류 계수 및 온도 검증
        if h_conv is None or h_conv < 0:
            raise ValueError(f"h_conv는 0 이상이어야 합니다. 현재 값: {h_conv}")
        if T_ambient is None or T_ambient < 0:
            raise ValueError(f"T_ambient는 0 이상이어야 합니다. 현재 값: {T_ambient}")
        
        t_start = data.get('t_start', 0)
        t_end = data.get('t_end', 1000.0)
        
        # 시간 범위 검증
        if t_start < 0:
            raise ValueError(f"t_start는 0 이상이어야 합니다. 현재 값: {t_start}")
        if t_end <= t_start:
            raise ValueError(f"t_end는 t_start보다 커야 합니다. t_start={t_start}, t_end={t_end}")
        # DoS 방지: 최대 시뮬레이션 시간 제한
        if t_end > MAX_T_END:
            raise ValueError(f"t_end는 {MAX_T_END:.0e} 초 이하여야 합니다. 현재 값: {t_end:.2e}")
        
        # 시간 포인트 수 계산 및 검증
        n_time_points = 50  # 기본값
        # 향후 동적 조정 시를 대비한 검증
        if n_time_points > MAX_T_EVAL_POINTS:
            raise ValueError(f"시간 포인트 수는 {MAX_T_EVAL_POINTS}개 이하여야 합니다. 현재 값: {n_time_points}")
        t_eval = np.logspace(np.log10(t_start + 1e-6), np.log10(t_end), n_time_points)
        
        # 2D 원통좌표계 그리드 설정
        # r 방향: 0부터 R_max까지 (소자 반지름보다 크게 설정)
        r_max_multiplier = data.get('r_max_multiplier', 10.0)  # 기본값 10
        
        # r_max_multiplier 검증 (1 이상 100 이하)
        if r_max_multiplier is None:
            r_max_multiplier = 10.0
        if r_max_multiplier < 1.0:
            raise ValueError(f"r_max_multiplier는 1 이상이어야 합니다. 현재 값: {r_max_multiplier}")
        if r_max_multiplier > 100.0:
            raise ValueError(f"r_max_multiplier는 100 이하여야 합니다. 현재 값: {r_max_multiplier}")
        
        R_max = device_radius_m * r_max_multiplier  # 소자 반지름의 배수만큼까지 계산
        Nr = data.get('Nr', 50)  # 반경 방향 그리드 수 (60 → 50으로 감소, 속도 향상)
        
        # 그리드 파라미터 검증
        if Nr < 3:
            raise ValueError(f"Nr은 최소 3 이상이어야 합니다. 현재 값: {Nr}")
        # DoS 방지: Nr 상한 검증
        if Nr > MAX_NR:
            raise ValueError(f"Nr은 {MAX_NR} 이하여야 합니다. 현재 값: {Nr}")
        if device_area_mm2 <= 0:
            raise ValueError(f"device_area_mm2는 0보다 커야 합니다. 현재 값: {device_area_mm2}")
        
        # z 방향: 레이어별 그리드 (두꺼운 층은 적은 포인트, 얇은 층은 많은 포인트)
        # 물리 일관성 유지: 두께는 원래대로, 그리드 포인트 수만 조절
        default_points_map = {
            'Glass': 8,      # 두꺼운 층: 적은 포인트 (coarse)
            'ITO': 12, 
            'HTL': 12, 
            'Perovskite': 25,  # 핵심 레이어: 많은 포인트 (fine)
            'ETL': 12, 
            'Cathode': 12, 
            'Resin': 6,      # 두꺼운 층: 적은 포인트 (coarse)
            'Heat sink': 6   # 두꺼운 층: 적은 포인트 (coarse)
        }
        points_per_layer = [default_points_map.get(name, 15) for name in layer_names]
        
        z_nodes = [0.0]
        layer_indices_map = []
        start_idx = 0
        num_layers = min(len(layer_names), len(thickness_layers), len(points_per_layer))
        
        debug_messages = []
        debug_messages.append("=== z 방향 그리드 생성 ===")
        flush_print(f"=== z 방향 그리드 생성 ===")
        for i in range(num_layers):
            thickness = thickness_layers[i]
            num_points = points_per_layer[i] if i < len(points_per_layer) else 15
            layer_nodes = np.linspace(z_nodes[-1], z_nodes[-1] + thickness, num_points + 1)
            z_nodes.extend(layer_nodes[1:])
            end_idx = start_idx + num_points
            layer_indices_map.append(slice(start_idx, end_idx + 1))
            start_idx = end_idx
        
        z = np.array(z_nodes)
        # 셀 두께 배열 명확히 정의: dz_cell[j] = z[j+1] - z[j] (j=0부터 Nz-2까지)
        # 경계 조건에서 사용할 실제 셀 두께
        dz_cell = z[1:] - z[:-1]  # 길이: Nz-1
        # 인터페이스 간격 배열 (라플라시안 계산용, 길이 Nz로 맞춤)
        dz = np.concatenate([dz_cell, [dz_cell[-1]]]) if len(dz_cell) > 0 else np.array([1e-9])
        Nz = len(z)
        
        # DoS 방지: Nz 상한 검증
        if Nz > MAX_NZ:
            raise ValueError(f"Nz는 {MAX_NZ} 이하여야 합니다. 현재 값: {Nz} (레이어 수: {len(layer_names)})")
        
        # r 방향 그리드 (0부터 R_max까지, r 근처에서 촘촘하게, 뒷쪽으로 갈수록 거칠게)
        # r=0부터 device_radius_m까지: 촘촘하게 (균일 그리드)
        Nr_fine = int(Nr * 0.6)  # 전체의 60%를 r 근처에 할당
        r_fine = np.linspace(0, device_radius_m, Nr_fine)
        
        # device_radius_m부터 R_max까지: 점점 더 거칠게 (로그 스케일 사용)
        Nr_coarse = Nr - Nr_fine + 1  # 나머지 그리드 수 (+1은 중복 제거용)
        # 로그 스케일로 점진적으로 증가하는 간격
        r_coarse_log = np.logspace(
            np.log10(device_radius_m + 1e-9),  # device_radius_m에서 시작 (0 방지)
            np.log10(R_max),
            Nr_coarse
        )
        # 첫 번째 점이 device_radius_m과 정확히 일치하도록 조정
        r_coarse_log[0] = device_radius_m
        
        # 두 구간 결합 (중복 제거)
        r = np.concatenate([r_fine, r_coarse_log[1:]])
        # 셀 두께 배열 명확히 정의: dr_cell[i] = r[i+1] - r[i] (i=0부터 Nr-2까지)
        # 경계 조건에서 사용할 실제 셀 두께
        dr_cell = r[1:] - r[:-1]  # 길이: Nr-1
        # 인터페이스 간격 배열 (라플라시안 계산용, 길이 Nr로 맞춤)
        # r=0에서 dr[0] 사용 (r=0은 특이점이므로 첫 번째 셀의 두께 사용)
        dr = np.concatenate([[r[1] - r[0]], dr_cell]) if len(dr_cell) > 0 else np.array([1e-9])
        Nr = len(r)
        
        # 2D 그리드 메시
        R, Z = np.meshgrid(r, z, indexing='ij')  # R[i,j], Z[i,j] = (r[i], z[j])
        
        # 물성 배열 (2D) - 등방성 열전도도 (압축 제거로 이방성 불필요)
        # 모든 레이어는 등방성: k_r = k_z = k_therm_layers[i]
        k_r_grid = np.zeros((Nr, Nz))  # r 방향 열전도도
        k_z_grid = np.zeros((Nr, Nz))  # z 방향 열전도도
        rho_cp_grid = np.zeros((Nr, Nz))
        
        num_layers_for_props = min(len(layer_indices_map), len(k_therm_layers), 
                                   len(rho_layers_effective), len(c_p_layers_effective))
        for i in range(num_layers_for_props):
            z_slice = layer_indices_map[i]
            
            # 모든 레이어는 등방성 (압축 제거로 이방성 불필요)
            k_r_grid[:, z_slice] = k_therm_layers[i]
            k_z_grid[:, z_slice] = k_therm_layers[i]
            rho_cp_grid[:, z_slice] = rho_layers_effective[i] * c_p_layers_effective[i]
        
        # 열원 위치 (Perovskite 레이어, r < device_radius_m 영역)
        try:
            perovskite_layer_index = layer_names.index('Perovskite')
        except ValueError:
            perovskite_layer_index = 1 if len(layer_names) > 1 else 0
        
        max_valid_index = min(len(layer_indices_map), len(thickness_layers), 
                             len(rho_layers), len(c_p_layers)) - 1
        if perovskite_layer_index > max_valid_index:
            perovskite_layer_index = max(0, max_valid_index)
        
        perovskite_z_slice = layer_indices_map[perovskite_layer_index]
        L_perovskite = thickness_layers[perovskite_layer_index]
        
        # Perovskite 두께 검증 (0 이하 체크)
        if L_perovskite <= 0:
            raise ValueError(f"Perovskite 레이어 두께가 0 이하입니다. L_perovskite = {L_perovskite} m")
        
        # 열원 마스크: r < device_radius_m이고 Perovskite 레이어인 영역
        source_mask = np.zeros((Nr, Nz), dtype=bool)
        for i in range(Nr):
            if r[i] < device_radius_m:
                source_mask[i, perovskite_z_slice] = True
        
        # 열원 강도 (W/m³)
        # Q_A는 W/m²이므로, Perovskite 두께로 나누어 W/m³로 변환
        Q_volumetric = Q_A / L_perovskite  # W/m³
        C_source_term = Q_volumetric / (rho_layers[perovskite_layer_index] * c_p_layers[perovskite_layer_index])
        
        # 디버깅: 열원 정보 출력
        num_source_nodes = np.sum(source_mask)
        flush_print(f"=== 열원 정보 ===")
        flush_print(f"Q_A = {Q_A:.2f} W/m²")
        flush_print(f"L_perovskite = {L_perovskite*1e9:.2f} nm")
        flush_print(f"Q_volumetric = {Q_volumetric:.2e} W/m³")
        flush_print(f"rho_cp = {rho_layers[perovskite_layer_index] * c_p_layers[perovskite_layer_index]:.2e} J/(m³·K)")
        flush_print(f"C_source_term = {C_source_term:.6e} K/s")
        flush_print(f"열원이 적용되는 노드 수: {num_source_nodes}개")
        flush_print(f"device_radius_m = {device_radius_m*1e3:.4f} mm")
        flush_print(f"r[0] = {r[0]*1e3:.4f} mm, r[-1] = {r[-1]*1e3:.4f} mm")
        
        # 초기 조건
        T0 = np.full((Nr, Nz), T_ambient)
        T0_flat = T0.flatten()
        N_total = Nr * Nz
        
        # DoS 방지: 총 노드 수 상한 검증
        if N_total > MAX_N_TOTAL:
            raise ValueError(f"총 노드 수(Nr * Nz)는 {MAX_N_TOTAL} 이하여야 합니다. 현재 값: {N_total} (Nr={Nr}, Nz={Nz})")
        
        # RHS 최적화: 사전 계산된 값들
        # 1. 열원 항 사전 계산 (flat 벡터)
        source_flat = np.zeros(N_total)
        source_flat[source_mask.ravel()] = C_source_term
        
        # 2. 경계 노드 인덱스 사전 계산 (Flat index)
        idx_z_bottom = np.arange(Nr) * Nz           # z=0
        idx_z_top = np.arange(Nr) * Nz + (Nz - 1)  # z=z_max
        idx_r_max = np.arange(Nz) + (Nr - 1) * Nz  # r=R_max
        
        # 3. 경계 조건 계산에 필요한 물성값 사전 추출
        rho_cp_bottom = rho_cp_grid[:, 0]           # z=0 경계
        rho_cp_top = rho_cp_grid[:, -1]            # z=z_max 경계
        rho_cp_r_max = rho_cp_grid[-1, :]          # r=R_max 경계
        # 경계 노드의 control volume 두께를 내부 노드와 일관되게 정의
        # 내부 노드 j의 control volume 두께: (dz_cell[j-1] + dz_cell[j]) / 2
        # 경계 노드도 동일한 방식으로 정의하여 보존성 유지
        if len(dz_cell) > 0:
            # z=0 (j=0): 첫 번째 셀의 두께만 사용 (경계면이 셀 시작점에 있음)
            dz_bottom = dz_cell[0]
            # z=z_max (j=Nz-1): 마지막 두 셀의 평균 (내부 노드와 동일한 방식)
            if len(dz_cell) > 1:
                dz_top = (dz_cell[-2] + dz_cell[-1]) * 0.5
            else:
                dz_top = dz_cell[-1]
        else:
            dz_bottom = 1e-9
            dz_top = 1e-9
        # r=R_max 경계 노드의 control volume 두께
        if len(dr_cell) > 0:
            if len(dr_cell) > 1:
                dr_r_max = (dr_cell[-2] + dr_cell[-1]) * 0.5  # 마지막 두 셀의 평균
            else:
                dr_r_max = dr_cell[-1]
        else:
            dr_r_max = 1e-9
        
        # 스파스 행렬 구성 (Numba 코어 + SciPy 래퍼)
        # _build_sparse_laplacian_core는 모듈 전역으로 정의되어 재컴파일 없이 재사용됨
        def build_sparse_laplacian():
            data, rows, cols, idx_count = _build_sparse_laplacian_core(
                Nr, Nz, N_total, r, dr_cell, dz_cell, k_r_grid, k_z_grid, rho_cp_grid
            )
            data = data[:idx_count]
            rows = rows[:idx_count]
            cols = cols[:idx_count]
            return sparse.csr_matrix((data, (rows, cols)), shape=(N_total, N_total))
        
        # 처음부터 CSR로 생성 (최적화)
        laplacian_csr = build_sparse_laplacian().tocsr()
        
        # CSR Matrix Template을 위한 대각 성분 인덱스 미리 계산
        # Jacobian에서 업데이트할 대각 성분의 인덱스를 미리 찾아둠
        diag_indices_dict = {}
        diag_data_indices_dict = {}  # CSR data 배열 인덱스
        
        # r=R_max 경계 인덱스 (측면) - 벡터화
        if Nr > 0:
            r_max_indices = np.arange(Nz) + (Nr - 1) * Nz
            diag_indices_dict['r_max'] = r_max_indices
            # CSR data 인덱스 찾기 (벡터화)
            r_max_data_indices = np.zeros(Nz, dtype=int)
            for j, idx in enumerate(r_max_indices):
                row_start = laplacian_csr.indptr[idx]
                row_end = laplacian_csr.indptr[idx + 1]
                row_indices = laplacian_csr.indices[row_start:row_end]
                diag_pos = np.where(row_indices == idx)[0]
                if len(diag_pos) > 0:
                    r_max_data_indices[j] = row_start + diag_pos[0]
                else:
                    r_max_data_indices[j] = -1  # 대각 성분 없음
            diag_data_indices_dict['r_max'] = r_max_data_indices
        
        # z=0 (하부) 경계 인덱스 - 벡터화
        z_bottom_indices = np.arange(Nr) * Nz
        diag_indices_dict['z_bottom'] = z_bottom_indices
        # CSR data 인덱스 찾기 (벡터화)
        z_bottom_data_indices = np.zeros(Nr, dtype=int)
        for i, idx in enumerate(z_bottom_indices):
            row_start = laplacian_csr.indptr[idx]
            row_end = laplacian_csr.indptr[idx + 1]
            row_indices = laplacian_csr.indices[row_start:row_end]
            diag_pos = np.where(row_indices == idx)[0]
            if len(diag_pos) > 0:
                z_bottom_data_indices[i] = row_start + diag_pos[0]
            else:
                z_bottom_data_indices[i] = -1  # 대각 성분 없음
        diag_data_indices_dict['z_bottom'] = z_bottom_data_indices
        
        # z=z_max (상부) 경계 인덱스 - 벡터화
        z_top_indices = np.arange(Nr) * Nz + (Nz - 1)
        diag_indices_dict['z_top'] = z_top_indices
        # CSR data 인덱스 찾기 (벡터화)
        z_top_data_indices = np.zeros(Nr, dtype=int)
        for i, idx in enumerate(z_top_indices):
            row_start = laplacian_csr.indptr[idx]
            row_end = laplacian_csr.indptr[idx + 1]
            row_indices = laplacian_csr.indices[row_start:row_end]
            diag_pos = np.where(row_indices == idx)[0]
            if len(diag_pos) > 0:
                z_top_data_indices[i] = row_start + diag_pos[0]
            else:
                z_top_data_indices[i] = -1  # 대각 성분 없음
        diag_data_indices_dict['z_top'] = z_top_data_indices
        
        # 진행 상황 추적을 위한 변수
        last_print_time = [t_start]  # 리스트로 감싸서 클로저에서 수정 가능하게
        
        # 진행률 업데이트 함수
        def update_progress(progress, message):
            with progress_lock:
                progress_store[session_id] = {
                    'progress': progress, 
                    'message': message,
                    'timestamp': time.time()  # 타임스탬프 추가
                }
        
        # 초기 진행률 설정
        update_progress(5, '그리드 생성 중...')
        
        # PDE 시스템 정의 (RHS 최적화: Flat indexing 사용)
        def pde_system(t, T_flat):
            # (1) 열 전도 항 계산 (Sparse Matrix-Vector Multiplication)
            # laplacian_csr.dot()은 내부적으로 최적화된 C/Fortran 루프를 사용합니다
            dTdt = laplacian_csr.dot(T_flat)
            
            # (2) 열원 항 더하기 (In-place 연산으로 임시 배열 생성 방지)
            dTdt += source_flat
            
            # 진행률 계산 (5% ~ 95%)
            progress_pct = 5 + (t - t_start) / (t_end - t_start) * 90
            progress_pct = min(95, max(5, progress_pct))
            
            # 주기적으로 진행 상황 출력 (매 5초마다, lock으로 안전하게 업데이트)
            # 단순화: progress_state_cache 제거, progress_store만 사용
            if t - last_print_time[0] >= 5.0:  # 5초마다 업데이트 (오버헤드 최소화)
                T_center = T_flat[0]  # r=0, z=0 (flat index)
                message = f"진행 중... t = {t:.3f} s ({t/t_end*100:.1f}%), T[0, 0] = {T_center:.2f} K"
                flush_print(message)
                # Lock으로 안전하게 progress_store 직접 업데이트
                with progress_lock:
                    progress_store[session_id] = {
                        'progress': progress_pct,
                        'message': message,
                        'timestamp': time.time()
                    }
                last_print_time[0] = t
            
            # 첫 시간 스텝 디버깅
            if t == t_start or abs(t - t_start) < 1e-6:
                T_center = T_flat[0]
                dTdt_source_val = source_flat[0] if source_flat[0] != 0 else 0.0
                dTdt_transport_val = dTdt[0] - source_flat[0]
                flush_print(f"=== 첫 시간 스텝 디버깅 (t={t}) ===")
                flush_print(f"T[0, 0] = {T_center:.2f} K")
                flush_print(f"dTdt_source[0] = {dTdt_source_val:.6f}")
                flush_print(f"dTdt_transport[0] = {dTdt_transport_val:.6f}")
                flush_print(f"laplacian_csr[0, 0] = {laplacian_csr[0, 0]:.6f}")
                if source_flat[0] != 0:
                    flush_print(f"열원 위치: source_flat[0] = {source_flat[0]:.6f}, C_source_term = {C_source_term:.6f}")
                flush_print(f"솔버 시작... (t_end = {t_end:.1f} s)")
                update_progress(10, f'솔버 시작... (t_end = {t_end:.1f} s)')
            
            # (3) 경계 플럭스 반영 (Flat indexing 사용, reshape 없이 직접 연산)
            # FVM에서 경계 노드의 control volume: 경계면이 셀 경계에 있으면
            # 경계 노드의 control volume 두께는 경계면에서 셀 중심까지의 거리 = 셀 두께 / 2
            # 하지만 dz_bottom = dz_cell[0]이므로, control volume 두께는 dz_bottom / 2가 맞음
            
            # z=0 (하부): 대류 + 방사
            T_bottom = T_flat[idx_z_bottom]
            # T ≈ T_ambient일 때 수치 안정성을 위해 방사 항을 더 정확하게 계산
            # (T^4 - T_ambient^4) = (T^2 + T_ambient^2)(T + T_ambient)(T - T_ambient)
            # T ≈ T_ambient일 때는 직접 계산보다 이렇게 인수분해하면 더 정확함
            T_bottom_diff = T_bottom - T_ambient
            T_bottom_sum = T_bottom + T_ambient
            T_bottom_sq_sum = T_bottom**2 + T_ambient**2
            # 수치 안정성: T ≈ T_ambient일 때 방사 항을 정확히 계산
            radiation_bottom = epsilon_bottom * sigma * T_bottom_sq_sum * T_bottom_sum * T_bottom_diff
            flux_bottom = h_conv * T_bottom_diff + radiation_bottom
            # 경계 노드의 control volume 두께: 경계면에서 셀 중심까지 = dz_bottom / 2
            # 하지만 실제 문제는 경계 조건이 너무 강하게 적용되어 열 손실이 과도할 수 있음
            # 따라서 경계 노드의 control volume 두께를 dz_bottom으로 사용 (더 약한 경계 조건)
            dTdt[idx_z_bottom] -= flux_bottom / (rho_cp_bottom * dz_bottom)
            
            # z=z_max (상부): 대류 + 방사
            T_top = T_flat[idx_z_top]
            T_top_diff = T_top - T_ambient
            T_top_sum = T_top + T_ambient
            T_top_sq_sum = T_top**2 + T_ambient**2
            radiation_top = epsilon_top * sigma * T_top_sq_sum * T_top_sum * T_top_diff
            flux_top = h_conv * T_top_diff + radiation_top
            # 경계 노드의 control volume 두께를 dz_top으로 사용
            dTdt[idx_z_top] -= flux_top / (rho_cp_top * dz_top)
            
            # r=R_max (측면): 대류 + 방사
            if Nr > 0:
                T_side = T_flat[idx_r_max]
                T_side_diff = T_side - T_ambient
                T_side_sum = T_side + T_ambient
                T_side_sq_sum = T_side**2 + T_ambient**2
                radiation_side = epsilon_side * sigma * T_side_sq_sum * T_side_sum * T_side_diff
                flux_side = h_conv * T_side_diff + radiation_side
                # 경계 노드의 control volume 두께를 dr_r_max로 사용
                dTdt[idx_r_max] -= flux_side / (rho_cp_r_max * dr_r_max)
            
            return dTdt
        
        # Jacobian (스파스 행렬) - 복사 열전달 항의 미분값 + 대류 항의 미분값 포함 (벡터화 및 CSR in-place 업데이트)
        # reshape 제거: flat indexing으로 직접 접근하여 메모리 복사 방지
        def jacobian(t, T_flat):
            """Jacobian 행렬: 라플라시안 + 복사 열전달 항의 미분값 (4εσT³) + 대류 항의 미분값 (h_conv)
            벡터화 및 CSR in-place 업데이트로 최적화됨 (data만 복사, 구조 재사용)
            reshape 제거: flat indexing으로 직접 접근"""
            # 구조 복사 없이 data 배열만 복사 (indices, indptr는 재사용)
            # 이렇게 하면 메모리 할당 비용을 크게 줄일 수 있음
            J_data = laplacian_csr.data.copy()
            
            # 복사 열전달 항의 미분값: d(εσ(T^4 - T_ambient^4))/dT
            # 인수분해 형태: εσ(T^2 + T_ambient^2)(T + T_ambient)(T - T_ambient)
            # 미분: εσ[4T^3 + 3T^2*T_ambient - T*T_ambient^2]
            # T ≈ T_ambient일 때: ≈ 4εσT_ambient^3 (수치 안정적)
            # 대류 항의 미분값: d(h_conv*(T - T_ambient))/dT = h_conv
            # 경계 조건에서만 적용되므로 대각 성분만 업데이트
            # reshape 없이 flat indexing으로 직접 접근
            
            # r=R_max 경계: 측면 방사율 + 대류 (완전 벡터화, flat indexing)
            if Nr > 0 and 'r_max' in diag_data_indices_dict:
                # Flat indexing으로 직접 접근 (reshape 없음)
                T_r_max = T_flat[idx_r_max]
                # 벡터화된 계산: 복사 항 미분 + 대류 항 미분
                # d/dT [εσ(T^4 - T_ambient^4)] = 4εσT^3
                # T_ambient는 상수이므로 미분하면 0이 됨
                # RHS에서 수치 안정성을 위해 인수분해 형태로 계산하더라도, 미분은 동일함
                radiation_deriv_r_max = 4.0 * epsilon_side * sigma * (T_r_max**3)
                # RHS와 일관성 유지: / 2 제거 (경계 노드의 control volume 두께를 dr_r_max로 사용)
                diag_values_r_max = (-radiation_deriv_r_max - h_conv) / (rho_cp_r_max * dr_r_max)
                # CSR data 배열 직접 수정 (벡터화)
                r_max_data_indices = diag_data_indices_dict['r_max']
                valid_mask = r_max_data_indices >= 0
                J_data[r_max_data_indices[valid_mask]] += diag_values_r_max[valid_mask]
            
            # z=0 (하부) 경계: 방사율 + 대류 (완전 벡터화, flat indexing)
            if 'z_bottom' in diag_data_indices_dict:
                # Flat indexing으로 직접 접근 (reshape 없음)
                T_z_bottom = T_flat[idx_z_bottom]
                # 벡터화된 계산: 복사 항 미분 + 대류 항 미분
                # d/dT [εσ(T^4 - T_ambient^4)] = 4εσT^3
                radiation_deriv_z_bottom = 4.0 * epsilon_bottom * sigma * (T_z_bottom**3)
                # RHS와 일관성 유지: / 2 제거 (경계 노드의 control volume 두께를 dz_bottom으로 사용)
                diag_values_z_bottom = (-radiation_deriv_z_bottom - h_conv) / (rho_cp_bottom * dz_bottom)
                # CSR data 배열 직접 수정 (벡터화)
                z_bottom_data_indices = diag_data_indices_dict['z_bottom']
                valid_mask = z_bottom_data_indices >= 0
                J_data[z_bottom_data_indices[valid_mask]] += diag_values_z_bottom[valid_mask]
            
            # z=z_max (상부) 경계: 방사율 + 대류 (완전 벡터화, flat indexing)
            if 'z_top' in diag_data_indices_dict:
                # Flat indexing으로 직접 접근 (reshape 없음)
                T_z_top = T_flat[idx_z_top]
                # 벡터화된 계산: 복사 항 미분 + 대류 항 미분
                # d/dT [εσ(T^4 - T_ambient^4)] = 4εσT^3
                radiation_deriv_z_top = 4.0 * epsilon_top * sigma * (T_z_top**3)
                # RHS와 일관성 유지: / 2 제거 (경계 노드의 control volume 두께를 dz_top으로 사용)
                diag_values_z_top = (-radiation_deriv_z_top - h_conv) / (rho_cp_top * dz_top)
                # CSR data 배열 직접 수정 (벡터화)
                z_top_data_indices = diag_data_indices_dict['z_top']
                valid_mask = z_top_data_indices >= 0
                J_data[z_top_data_indices[valid_mask]] += diag_values_z_top[valid_mask]
            
            # 새로운 CSR 객체 생성 (indices, indptr는 재사용하여 메모리 효율 극대화)
            return sparse.csr_matrix(
                (J_data, laplacian_csr.indices, laplacian_csr.indptr), 
                shape=laplacian_csr.shape
            )
        
        # 솔버 실행 (허용 오차 완화로 속도 향상)
        flush_print(f"=== 솔버 실행 시작 ===")
        flush_print(f"그리드 크기: {Nr} x {Nz} = {Nr*Nz} 노드")
        flush_print(f"시간 범위: {t_start:.6f} ~ {t_end:.1f} s")
        flush_print(f"시간 포인트 수: {len(t_eval)}")
        
        update_progress(15, f'솔버 실행 중... (그리드: {Nr}x{Nz})')
        
        start_time = time.time()
        
        # Jacobian 사용 여부 선택 (성능 비교용)
        # True: 명시적 Jacobian 사용 (정확하지만 느릴 수 있음)
        # False: 수치 Jacobian 사용 (BDF 내부 추정, 빠를 수 있음)
        use_explicit_jacobian = True  # 성능 테스트 시 False로 변경 가능
        
        try:
            solver_kwargs = {
                'fun': pde_system,
                't_span': [t_start, t_end],
                'y0': T0_flat,
                't_eval': t_eval,
                'method': 'BDF',
                'atol': 1e-6,  # 절대 오차 허용 범위 (더 엄격하게 설정)
                'rtol': 1e-4   # 상대 오차 허용 범위 (더 엄격하게 설정)
            }
            
            # Jacobian 사용 여부에 따라 조건부 추가
            if use_explicit_jacobian:
                solver_kwargs['jac'] = jacobian
                flush_print("=== 명시적 Jacobian 사용 ===")
            else:
                flush_print("=== 수치 Jacobian 사용 (BDF 내부 추정) ===")
            
            sol = solve_ivp(**solver_kwargs)
        except Exception as solver_error:
            error_msg = f"솔버 실행 중 오류: {str(solver_error)}"
            flush_print(f"❌ {error_msg}")
            flush_print(f"트레이스백:\n{traceback.format_exc()}")
            update_progress(0, f'솔버 오류: {str(solver_error)}')
            raise ValueError(error_msg) from solver_error
        
        elapsed_time = time.time() - start_time
        flush_print(f"=== 솔버 완료 (소요 시간: {elapsed_time:.1f} 초) ===")
        update_progress(95, f'결과 처리 중... (소요 시간: {elapsed_time:.1f} 초)')
        
        if not sol.success:
            error_msg = f"솔버 실패: {sol.message}"
            flush_print(f"❌ {error_msg}")
            update_progress(0, f'솔버 실패: {sol.message}')
            raise ValueError(error_msg)
        
        # 결과 처리
        T_result = sol.y.reshape(Nr, Nz, -1)  # (Nr, Nz, n_time)
        
        # 디버깅: 온도 값 확인
        flush_print(f"=== 솔버 결과 디버깅 ===")
        flush_print(f"솔버 성공: {sol.success}")
        flush_print(f"T_result shape: {T_result.shape}")
        flush_print(f"T_result min: {np.min(T_result):.2f} K, max: {np.max(T_result):.2f} K, mean: {np.mean(T_result):.2f} K")
        flush_print(f"T_ambient: {T_ambient:.2f} K ({T_ambient - 273.15:.2f} °C)")
        flush_print(f"초기 온도 T0[0, 0]: {T0[0, 0]:.2f} K")
        flush_print(f"최종 온도 T_result[0, 0, 0]: {T_result[0, 0, 0]:.2f} K ({T_result[0, 0, 0] - 273.15:.2f} °C)")
        flush_print(f"최종 온도 T_result[0, 0, -1]: {T_result[0, 0, -1]:.2f} K ({T_result[0, 0, -1] - 273.15:.2f} °C)")
        
        # 온도가 비정상적으로 낮은지 확인
        if np.min(T_result) < 100:
            flush_print(f"⚠️ 경고: 최소 온도가 100K 미만입니다! ({np.min(T_result):.2f} K)")
        if np.max(T_result) < T_ambient:
            flush_print(f"⚠️ 경고: 최대 온도가 주변 온도보다 낮습니다!")
        
        # 압축 제거로 z 좌표 복원 불필요 (원래 두께 사용)
        z_nm = z * 1e9
        r_mm = r * 1e3  # m를 mm로 변환
        
        # Glass와 ITO 경계점 찾기 (안전성 체크)
        if len(layer_indices_map) == 0:
            raise ValueError("레이어 정보가 없습니다.")
        
        glass_ito_boundary_idx = layer_indices_map[0].stop - 1
        if glass_ito_boundary_idx < 0 or glass_ito_boundary_idx >= len(z_nm):
            raise ValueError(f"Glass-ITO 경계 인덱스 오류: {glass_ito_boundary_idx}, z 길이: {len(z_nm)}")
        
        glass_ito_boundary_nm = z_nm[glass_ito_boundary_idx]
        active_start_idx = glass_ito_boundary_idx + 1
        
        # 활성층 위치 (ITO 시작점을 z=0으로)
        if active_start_idx >= len(z_nm):
            active_start_idx = len(z_nm) - 1
        position_active_nm = (z_nm[active_start_idx:] - glass_ito_boundary_nm).tolist()
        
        # 2D 온도 데이터 (최종 시간) - 안전성 체크
        final_time_idx = -1
        if T_result.shape[0] == 0 or T_result.shape[1] == 0 or T_result.shape[2] == 0:
            raise ValueError(f"온도 데이터 크기 오류: {T_result.shape}")
        
        if active_start_idx >= T_result.shape[1]:
            active_start_idx = T_result.shape[1] - 1
        
        # temperature_2d 다운샘플링 (메모리 절약: 최대 200x200)
        T_2d_raw = T_result[:, active_start_idx:, final_time_idx]  # (Nr, Nz_active)
        max_r_points = 200
        max_z_points = 200
        
        # 다운샘플링 필요 여부 확인
        if T_2d_raw.shape[0] > max_r_points or T_2d_raw.shape[1] > max_z_points:
            flush_print(f"⚠️ temperature_2d 다운샘플링: {T_2d_raw.shape} → 최대 ({max_r_points}, {max_z_points})")
            # 균등 간격으로 다운샘플링
            r_indices = np.linspace(0, T_2d_raw.shape[0] - 1, min(max_r_points, T_2d_raw.shape[0]), dtype=int)
            z_indices_2d = np.linspace(0, T_2d_raw.shape[1] - 1, min(max_z_points, T_2d_raw.shape[1]), dtype=int)
            T_2d_downsampled = T_2d_raw[np.ix_(r_indices, z_indices_2d)]
            temperature_2d = T_2d_downsampled.tolist()
            # 다운샘플링된 r, z 좌표도 저장
            r_mm_downsampled = r_mm[r_indices].tolist()
            position_active_nm_downsampled = [position_active_nm[i] for i in z_indices_2d]
        else:
            temperature_2d = T_2d_raw.tolist()
            r_mm_downsampled = r_mm.tolist()
            position_active_nm_downsampled = position_active_nm
        
        # 활성층 레이어 경계 (temperature_center 샘플링에 필요)
        active_layer_boundaries_nm = [0.0]
        try:
            max_idx = min(len(layer_names), len(thickness_layers_nm_original))
            for i in range(1, max_idx):
                if i >= len(thickness_layers_nm_original):
                    raise IndexError(f"인덱스 오류: i={i}")
                active_layer_boundaries_nm.append(float(active_layer_boundaries_nm[-1] + thickness_layers_nm_original[i]))
        except (IndexError, ValueError) as e:
            raise ValueError(f"레이어 경계 계산 오류: {str(e)}") from e
        
        # r=0에서의 온도 프로파일 (z 방향, 메모리 최적화: 샘플링)
        # 전체 데이터는 매우 크므로 (Nz_active × n_time), 중요한 지점만 전달
        time_indices_sampled = []  # 스코프를 위해 미리 정의
        if T_result.shape[0] > 0:
            # z 방향: 레이어 경계와 중심 지점만 샘플링
            # 시간 방향: 로그 스케일로 샘플링 (초기에는 촘촘, 후반에는 성글게)
            n_z_samples = min(50, T_result.shape[1] - active_start_idx)  # 최대 50개 z 위치
            n_time_samples = min(30, T_result.shape[2])  # 최대 30개 시간 포인트
            
            # z 방향 샘플링: 레이어 경계와 중심 포함
            z_indices_sampled = []
            if len(active_layer_boundaries_nm) > 1:
                # 레이어 경계 지점 찾기
                for i, boundary_nm in enumerate(active_layer_boundaries_nm):
                    if i < len(position_active_nm):
                        # 경계에 가장 가까운 인덱스 찾기
                        closest_idx = np.argmin(np.abs(np.array(position_active_nm) - boundary_nm))
                        if closest_idx not in z_indices_sampled:
                            z_indices_sampled.append(closest_idx)
                # 각 레이어의 중간 지점도 추가
                for i in range(len(active_layer_boundaries_nm) - 1):
                    mid_nm = (active_layer_boundaries_nm[i] + active_layer_boundaries_nm[i + 1]) / 2
                    closest_idx = np.argmin(np.abs(np.array(position_active_nm) - mid_nm))
                    if closest_idx not in z_indices_sampled:
                        z_indices_sampled.append(closest_idx)
            
            # 샘플링이 부족하면 균등 분포로 보완
            if len(z_indices_sampled) < n_z_samples:
                z_indices_sampled = sorted(set(z_indices_sampled))
                remaining = n_z_samples - len(z_indices_sampled)
                if remaining > 0:
                    all_indices = set(range(T_result.shape[1] - active_start_idx))
                    available = sorted(all_indices - set(z_indices_sampled))
                    step = max(1, len(available) // remaining)
                    z_indices_sampled.extend(available[::step])
                    z_indices_sampled = sorted(set(z_indices_sampled))[:n_z_samples]
            
            # 시간 방향 샘플링: 로그 스케일
            if T_result.shape[2] > n_time_samples:
                time_indices_sampled = np.unique(
                    np.logspace(0, np.log10(T_result.shape[2] - 1), n_time_samples, dtype=int)
                ).tolist()
            else:
                time_indices_sampled = list(range(T_result.shape[2]))
            
            # 샘플링된 데이터만 전달
            temperature_center_sampled = []
            for z_idx in z_indices_sampled:
                z_idx_actual = active_start_idx + z_idx
                if z_idx_actual < T_result.shape[1]:
                    temp_profile = T_result[0, z_idx_actual, :][time_indices_sampled].tolist()
                    temperature_center_sampled.append({
                        'z_index': int(z_idx),
                        'position_nm': float(position_active_nm[z_idx]),
                        'temperature': temp_profile,
                        'time_indices': time_indices_sampled
                    })
            
            temperature_center = temperature_center_sampled
        else:
            temperature_center = []
            time_indices_sampled = []
        
        # 페로브스카이트 중간 지점에서의 시간에 따른 온도 (r=0)
        perovskite_mid_idx = None
        if perovskite_layer_index < len(layer_indices_map):
            perovskite_start_idx = layer_indices_map[perovskite_layer_index].start
            perovskite_end_idx = layer_indices_map[perovskite_layer_index].stop
            perovskite_mid_idx = (perovskite_start_idx + perovskite_end_idx) // 2
            
            # 디버깅 정보
            print(f"perovskite_layer_index: {perovskite_layer_index}")
            print(f"perovskite_start_idx: {perovskite_start_idx}, perovskite_end_idx: {perovskite_end_idx}")
            print(f"perovskite_mid_idx: {perovskite_mid_idx}, T_result.shape[1]: {T_result.shape[1]}")
            
            if perovskite_mid_idx < T_result.shape[1] and perovskite_mid_idx >= 0:
                perovskite_center_temp = T_result[0, perovskite_mid_idx, :].tolist()
                print(f"perovskite_center_temp (first 5): {perovskite_center_temp[:5] if len(perovskite_center_temp) > 5 else perovskite_center_temp}")
            else:
                # 안전한 대체: Perovskite 레이어 내의 유효한 인덱스 사용
                if perovskite_start_idx < T_result.shape[1] and perovskite_start_idx >= 0:
                    perovskite_center_temp = T_result[0, perovskite_start_idx, :].tolist()
                    perovskite_mid_idx = perovskite_start_idx
                    print(f"Using perovskite_start_idx instead: {perovskite_start_idx}")
                else:
                    perovskite_mid_idx = max(0, min(T_result.shape[1]-1, Nz//2))
                    perovskite_center_temp = T_result[0, perovskite_mid_idx, :].tolist() if T_result.shape[1] > 0 else []
                    print(f"Using fallback index")
        else:
            # perovskite_layer_index가 유효하지 않은 경우 중간 인덱스 사용
            perovskite_mid_idx = T_result.shape[1] // 2 if T_result.shape[1] > 0 else 0
            perovskite_center_temp = T_result[0, perovskite_mid_idx, :].tolist() if T_result.shape[1] > 0 else []
            print(f"perovskite_layer_index invalid, using mid_z_idx: {perovskite_mid_idx}")
        
        # 세 가지 프로파일 계산
        # 1. r=0에서 z에 따른 최종온도 프로파일
        final_time_idx = -1
        temp_profile_r0_z = T_result[0, :, final_time_idx].tolist()  # r=0 (인덱스 0), 모든 z, 최종 시간
        z_profile_nm = z_nm.tolist()  # 전체 z 좌표 (nm)
        
        # r=0에서 z, time에 따른 전체 온도 데이터 (Sheet1용)
        # 메모리 절약: 다운샘플링 (최대 500개 z 포인트, 모든 시간 포인트)
        temp_profile_r0_z_time = None
        if T_result.shape[1] > 500:
            # z 방향 다운샘플링
            z_indices = np.linspace(0, T_result.shape[1] - 1, 500, dtype=int)
            temp_profile_r0_z_time = T_result[0, z_indices, :].tolist()  # (500, n_time)
            z_profile_nm_sampled = [float(z_profile_nm[i]) for i in z_indices]  # 리스트로 변환
        else:
            temp_profile_r0_z_time = T_result[0, :, :].tolist()  # (Nz, n_time)
            z_profile_nm_sampled = [float(z) for z in z_profile_nm]  # 리스트로 변환
        
        flush_print(f"=== 프로파일 1: r=0에서 z에 따른 최종온도 ===")
        flush_print(f"데이터 크기: {len(temp_profile_r0_z)}개 z 포인트")
        flush_print(f"온도 범위: {min(temp_profile_r0_z):.2f} ~ {max(temp_profile_r0_z):.2f} K")
        flush_print(f"=== r=0에서 z, time에 따른 온도 데이터 (Sheet1용) ===")
        flush_print(f"데이터 크기: {len(temp_profile_r0_z_time)}개 z 포인트 x {len(temp_profile_r0_z_time[0]) if temp_profile_r0_z_time and len(temp_profile_r0_z_time) > 0 else 0}개 시간 포인트")
        
        # 2. z=perovskite 중점에서 r에 따른 최종온도 프로파일
        if perovskite_mid_idx is not None and perovskite_mid_idx < T_result.shape[1] and perovskite_mid_idx >= 0:
            temp_profile_z_perovskite_r = T_result[:, perovskite_mid_idx, final_time_idx].tolist()  # 모든 r, z=perovskite 중점, 최종 시간
            flush_print(f"=== 프로파일 2: z=perovskite 중점에서 r에 따른 최종온도 ===")
            flush_print(f"perovskite_mid_idx: {perovskite_mid_idx}, z 좌표: {z_nm[perovskite_mid_idx]:.2f} nm")
            flush_print(f"데이터 크기: {len(temp_profile_z_perovskite_r)}개 r 포인트")
            flush_print(f"온도 범위: {min(temp_profile_z_perovskite_r):.2f} ~ {max(temp_profile_z_perovskite_r):.2f} K")
        else:
            temp_profile_z_perovskite_r = []
            flush_print(f"⚠️ 경고: perovskite_mid_idx가 유효하지 않습니다. 프로파일 2를 계산할 수 없습니다.")
        
        # 3. z=perovskite 중점, r=0에서 시간에 따른 온도 프로파일 (이미 계산됨: perovskite_center_temp)
        flush_print(f"=== 프로파일 3: z=perovskite 중점, r=0에서 시간에 따른 온도 ===")
        flush_print(f"데이터 크기: {len(perovskite_center_temp)}개 시간 포인트")
        if len(perovskite_center_temp) > 0:
            flush_print(f"온도 범위: {min(perovskite_center_temp):.2f} ~ {max(perovskite_center_temp):.2f} K")
        
        
        # NumPy 타입 변환
        def convert_to_python_type(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_python_type(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_type(item) for item in obj]
            else:
                return obj
        
        # 디버깅: 반환 전 온도 값 확인
        flush_print(f"=== 반환 데이터 확인 ===")
        if len(perovskite_center_temp) > 0:
            flush_print(f"perovskite_center_temp[0]: {perovskite_center_temp[0]:.2f} K ({perovskite_center_temp[0] - 273.15:.2f} °C)")
            flush_print(f"perovskite_center_temp[-1]: {perovskite_center_temp[-1]:.2f} K ({perovskite_center_temp[-1] - 273.15:.2f} °C)")
        if temperature_2d and len(temperature_2d) > 0 and len(temperature_2d[0]) > 0:
            flush_print(f"temperature_2d[0][0]: {temperature_2d[0][0]:.2f} K ({temperature_2d[0][0] - 273.15:.2f} °C)")
        
        try:
            update_progress(100, '완료!')
            
            # 결과 데이터 준비 (디스크 저장용)
            result_data = {
                'success': True,
                'session_id': session_id,
                'time': sol.t,  # NumPy 배열로 저장 (JSON 변환 전)
                'position_active_nm': np.array(position_active_nm_downsampled),  # 다운샘플링된 z 좌표
                'temperature_2d': np.array(temperature_2d),  # 다운샘플링된 데이터
                'temperature_center': temperature_center,  # 이미 리스트
                'r_mm': np.array(r_mm_downsampled),  # 다운샘플링된 r 좌표
                'perovskite_center_temp': np.array(perovskite_center_temp),
                'layer_boundaries_nm': np.array(active_layer_boundaries_nm),
                'layer_names': layer_names[1:] if len(layer_names) > 1 else [],
                'glass_ito_boundary_nm': glass_ito_boundary_nm,
                'device_radius_mm': device_radius_m * 1e3,
                'temp_profile_r0_z': np.array(temp_profile_r0_z),
                'z_profile_nm': np.array(z_profile_nm),
                'temp_profile_z_perovskite_r': np.array(temp_profile_z_perovskite_r),
                'perovskite_mid_z_nm': z_nm[perovskite_mid_idx] if perovskite_mid_idx is not None and perovskite_mid_idx < len(z_nm) else None
            }
            
            # 결과를 디스크에 저장 (npz 형식)
            result_file = os.path.join(RESULTS_DIR, f"{session_id}.npz")
            np.savez_compressed(result_file, **result_data)
            flush_print(f"=== 결과가 디스크에 저장되었습니다: {result_file} ===")
            
            # JSON 응답용 경량 데이터 (프론트엔드 표시용)
            result_summary = {
                'success': True,
                'session_id': session_id,
                'time': convert_to_python_type(sol.t.tolist()),
                'position_active_nm': convert_to_python_type(position_active_nm_downsampled),  # 다운샘플링된 z 좌표
                'temperature_2d': convert_to_python_type(temperature_2d),  # 다운샘플링된 데이터
                'temperature_center': convert_to_python_type(temperature_center),
                'r_mm': convert_to_python_type(r_mm_downsampled),
                'perovskite_center_temp': convert_to_python_type(perovskite_center_temp),
                'layer_boundaries_nm': convert_to_python_type(active_layer_boundaries_nm),
                'layer_names': layer_names[1:] if len(layer_names) > 1 else [],
                'glass_ito_boundary_nm': float(glass_ito_boundary_nm),
                'device_radius_mm': float(device_radius_m * 1e3),
                'temp_profile_r0_z': convert_to_python_type(temp_profile_r0_z),
                'z_profile_nm': convert_to_python_type(z_profile_nm),
                'temp_profile_r0_z_time': convert_to_python_type(temp_profile_r0_z_time) if temp_profile_r0_z_time is not None else None,  # r=0에서 z, time에 따른 온도 (Sheet1용)
                'z_profile_nm_sampled': convert_to_python_type(z_profile_nm_sampled) if temp_profile_r0_z_time is not None else None,  # 샘플링된 z 좌표
                'temp_profile_z_perovskite_r': convert_to_python_type(temp_profile_z_perovskite_r),
                'perovskite_mid_z_nm': float(z_nm[perovskite_mid_idx]) if perovskite_mid_idx is not None and perovskite_mid_idx < len(z_nm) else None
            }
            
            flush_print(f"=== 결과 반환 준비 완료 ===")
            flush_print(f"결과 크기: time={len(result_summary.get('time', []))}, temperature_2d shape={len(result_summary.get('temperature_2d', []))}x{len(result_summary.get('temperature_2d', [])[0]) if result_summary.get('temperature_2d') else 0}")
            flush_print(f"temperature_center 샘플링: {len(temperature_center)}개 z 위치, 각 {len(time_indices_sampled) if 'time_indices' in str(temperature_center) else '전체'}개 시간 포인트")
            
            # 결과를 progress_store에 저장
            # result_summary를 저장하여 프론트엔드에서 바로 사용 가능하도록 함
            with progress_lock:
                progress_store[session_id] = {
                    'progress': 100,
                    'message': '완료!',
                    'result_path': result_file,  # 디스크 경로도 저장 (백업용)
                    'result': result_summary,  # JSON 형식의 결과 데이터 저장
                    # 최소한의 메타데이터도 저장 (호환성 유지)
                    'result_metadata': {
                        'success': True,
                        'session_id': session_id,
                        'grid_size': f"{Nr}x{Nz}",
                        'time_points': len(sol.t),
                        'device_radius_mm': float(device_radius_m * 1e3),
                        'glass_ito_boundary_nm': float(glass_ito_boundary_nm),
                        'perovskite_mid_z_nm': float(z_nm[perovskite_mid_idx]) if perovskite_mid_idx is not None and perovskite_mid_idx < len(z_nm) else None
                    },
                    'timestamp': time.time()
                }
            
            flush_print(f"=== 결과가 progress_store에 저장되었습니다 ===")
        except Exception as result_error:
            error_msg = f"결과 처리 중 오류: {str(result_error)}"
            flush_print(f"❌ {error_msg}")
            flush_print(f"트레이스백:\n{traceback.format_exc()}")
            update_progress(0, f'결과 처리 오류: {str(result_error)}')
            raise
        
    except Exception as e:
        # 오류 발생 시 진행률 업데이트
        # 백그라운드 스레드에서는 jsonify를 반환하지 않고 progress_store에만 저장
        # 프론트엔드는 /api/progress/<session_id>로 에러 상태를 확인
        error_traceback = traceback.format_exc()
        flush_print(f"=== 시뮬레이션 오류 ===")
        flush_print(f"오류 타입: {type(e).__name__}")
        flush_print(f"오류 메시지: {str(e)}")
        flush_print(f"트레이스백:\n{error_traceback}")
        
        error_info = {
            'error_type': type(e).__name__,
            'error_message': str(e),
            'traceback': error_traceback
        }
        with progress_lock:
            progress_store[session_id] = {
                'progress': 0,
                'message': f'오류 발생: {str(e)}',
                'error': str(e),
                'error_details': error_info,
                'timestamp': time.time()
            }

if __name__ == '__main__':
    import sys
    # fly.io나 다른 클라우드 플랫폼에서 PORT 환경 변수 사용
    # 환경 변수가 없으면 명령줄 인자, 그래도 없으면 기본값 5000 사용
    port = int(os.environ.get('PORT', sys.argv[1] if len(sys.argv) > 1 else '5000'))
    
    # 운영 안정성: debug=False, use_reloader=False
    # debug=True는 reloader가 프로세스를 2개 띄울 수 있어 스레딩과 충돌 가능
    # 운영 환경에서는 gunicorn/uwsgi + Celery/RQ 패턴 권장
    DEBUG_MODE = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    USE_RELOADER = os.environ.get('FLASK_USE_RELOADER', 'False').lower() == 'true'
    
    # fly.io나 클라우드 환경에서는 모든 인터페이스에서 리슨해야 함
    host = os.environ.get('HOST', '0.0.0.0')
    
    try:
        app.run(debug=DEBUG_MODE, use_reloader=USE_RELOADER, port=port, host=host)
    except OSError as e:
        if 'Address already in use' in str(e) or 'Port already in use' in str(e):
            print(f"⚠️ 포트 {port}가 이미 사용 중입니다.")
            print(f"다른 포트(5001)로 시도합니다...")
            app.run(debug=DEBUG_MODE, use_reloader=USE_RELOADER, port=5001, host=host)
        else:
            raise
