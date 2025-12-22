from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from scipy.integrate import solve_ivp
import json
import traceback

app = Flask(__name__)
CORS(app)

@app.route('/')
def health():
    return jsonify({'status': 'ok', 'message': 'Flask backend is running'})

@app.route('/api/simulate', methods=['POST'])
def simulate():
    try:
        data = request.json
        
        # 파라미터 추출
        layer_names = data['layer_names']
        k_therm_layers_original = np.array(data['k_therm_layers'])
        rho_layers = np.array(data['rho_layers'])
        c_p_layers = np.array(data['c_p_layers'])
        thickness_layers_nm_original = np.array(data['thickness_layers_nm'])
        
        # Glass 두께를 10000배 줄여서 계산량 감소
        glass_thickness_scale_factor = 10000.0
        thickness_layers_nm = thickness_layers_nm_original.copy()
        if len(thickness_layers_nm_original) > 0:
            thickness_layers_nm[0] = thickness_layers_nm_original[0] / glass_thickness_scale_factor  # Glass만 10000배 얇게
        
        # Heat spreader와 Heat sink를 100nm로 압축
        target_thickness_nm = 100.0  # 목표 두께 (nm)
        heat_spreader_index = None
        heat_sink_index = None
        heat_spreader_scale_factor = None
        heat_sink_scale_factor = None
        
        for i, name in enumerate(layer_names):
            if name == 'Heat spreader' and i < len(thickness_layers_nm_original):
                heat_spreader_index = i
                if thickness_layers_nm_original[i] > target_thickness_nm:
                    heat_spreader_scale_factor = thickness_layers_nm_original[i] / target_thickness_nm
                    thickness_layers_nm[i] = target_thickness_nm
            elif name == 'Heat sink' and i < len(thickness_layers_nm_original):
                heat_sink_index = i
                if thickness_layers_nm_original[i] > target_thickness_nm:
                    heat_sink_scale_factor = thickness_layers_nm_original[i] / target_thickness_nm
                    thickness_layers_nm[i] = target_thickness_nm
        
        thickness_layers = thickness_layers_nm * 1e-9
        
        # Glass의 effective 물성 계산
        # 1. Thermal conductivity (열 저항 유지: R = L/k)
        k_therm_layers = k_therm_layers_original.copy()
        if len(k_therm_layers_original) > 0:
            k_therm_layers[0] = k_therm_layers_original[0] * glass_thickness_scale_factor  # Glass만 10000배 증가
        
        # Heat spreader와 Heat sink의 effective 물성 계산
        if heat_spreader_index is not None and heat_spreader_scale_factor is not None:
            k_therm_layers[heat_spreader_index] = k_therm_layers_original[heat_spreader_index] * heat_spreader_scale_factor
        if heat_sink_index is not None and heat_sink_scale_factor is not None:
            k_therm_layers[heat_sink_index] = k_therm_layers_original[heat_sink_index] * heat_sink_scale_factor
        
        # 2. Density와 Heat capacity (열용량 및 시간 상수 유지)
        rho_layers_effective = rho_layers.copy()
        c_p_layers_effective = c_p_layers.copy()
        if len(rho_layers) > 0:
            rho_layers_effective[0] = rho_layers[0] * glass_thickness_scale_factor  # Glass만 10000배 증가
        if heat_spreader_index is not None and heat_spreader_scale_factor is not None:
            rho_layers_effective[heat_spreader_index] = rho_layers[heat_spreader_index] * heat_spreader_scale_factor
        if heat_sink_index is not None and heat_sink_scale_factor is not None:
            rho_layers_effective[heat_sink_index] = rho_layers[heat_sink_index] * heat_sink_scale_factor
        
        voltage = data['voltage']
        current_density = data['current_density']
        eqe = data.get('eqe', 0.2)  # 기본값 20%
        # EQE를 고려한 실제 Joule heating: Q_effective = Q_A * (1 - EQE)
        Q_A = voltage * current_density * (1 - eqe)
        
        epsilon_top = data['epsilon_top']
        epsilon_bottom = data['epsilon_bottom']
        sigma = 5.67e-8
        h_conv = data['h_conv']
        T_ambient = data['T_ambient']
        
        t_start = data.get('t_start', 0)
        t_end = data.get('t_end', 1000.0)
        t_eval = np.linspace(t_start, t_end, 200)
        
        # 비균일 그리드 및 물성 배열 생성
        # 기본 points_per_layer 정의 (Glass, ITO, HTL, Perovskite, ETL, Cathode, Heat spreader, Heat sink 순서)
        default_points_map = {
            'Glass': 50, 'ITO': 20, 'HTL': 20, 'Perovskite': 40, 
            'ETL': 20, 'Cathode': 20, 'Heat spreader': 20, 'Heat sink': 30
        }
        # 레이어별로 points 수 할당
        points_per_layer = [default_points_map.get(name, 20) for name in layer_names]
        x_nodes = [0.0]
        layer_indices_map = []
        start_idx = 0
        # layer_names와 thickness_layers의 길이가 같아야 하지만, 안전을 위해 더 작은 길이 사용
        num_layers = min(len(layer_names), len(thickness_layers), len(points_per_layer))
        for i in range(num_layers):
            thickness = thickness_layers[i]
            num_points = points_per_layer[i] if i < len(points_per_layer) else 20
            layer_nodes = np.linspace(x_nodes[-1], x_nodes[-1] + thickness, num_points + 1)
            x_nodes.extend(layer_nodes[1:])
            end_idx = start_idx + num_points
            layer_indices_map.append(slice(start_idx, end_idx + 1))
            start_idx = end_idx
        x = np.array(x_nodes)
        dx = x[1:] - x[:-1]
        Nx = len(x)
        
        k_grid = np.zeros(Nx)
        rho_c_p_grid = np.zeros(Nx)
        # layer_indices_map의 길이와 물성 배열의 길이가 같아야 하지만, 안전을 위해 더 작은 길이 사용
        num_layers_for_props = min(len(layer_indices_map), len(k_therm_layers), len(rho_layers_effective), len(c_p_layers_effective))
        for i in range(num_layers_for_props):
            prop_slice = layer_indices_map[i]
            if i < len(k_therm_layers):
                k_grid[prop_slice] = k_therm_layers[i]
            if i < len(rho_layers_effective) and i < len(c_p_layers_effective):
                rho_c_p_grid[prop_slice] = rho_layers_effective[i] * c_p_layers_effective[i]
        
        # 열원 위치 계산 (Perovskite 레이어 찾기)
        try:
            perovskite_layer_index = layer_names.index('Perovskite')
        except ValueError:
            # Perovskite 레이어가 없는 경우 첫 번째 활성 레이어 사용 (Glass 제외)
            perovskite_layer_index = 1 if len(layer_names) > 1 else 0
        
        # 인덱스 범위 확인 및 조정
        max_valid_index = min(len(layer_indices_map), len(thickness_layers), len(rho_layers), len(c_p_layers)) - 1
        if perovskite_layer_index > max_valid_index:
            perovskite_layer_index = max(0, max_valid_index)
        
        perovskite_slice = layer_indices_map[perovskite_layer_index]
        L_perovskite = thickness_layers[perovskite_layer_index]
        C_source_term = Q_A / (L_perovskite * rho_layers[perovskite_layer_index] * c_p_layers[perovskite_layer_index])
        
        T0 = np.full(Nx, T_ambient)
        
        # PDE 시스템 정의
        def pde_system(t, T):
            dTdt_source = np.zeros_like(T)
            dTdt_transport = np.zeros_like(T)
            dTdt_source[perovskite_slice] = C_source_term
            k_interface = 2 * k_grid[:-1] * k_grid[1:] / (k_grid[:-1] + k_grid[1:])
            flux = -k_interface * (T[1:] - T[:-1]) / dx
            control_volume_widths = (dx[:-1] + dx[1:]) / 2
            dTdt_transport[1:-1] = (flux[:-1] - flux[1:]) / (control_volume_widths * rho_c_p_grid[1:-1])
            flux_out_bottom = h_conv * (T[0] - T_ambient) + epsilon_bottom * sigma * (T[0]**4 - T_ambient**4)
            dTdt_transport[0] = (-flux[0] - flux_out_bottom) / (rho_c_p_grid[0] * (dx[0]/2))
            flux_out_top = h_conv * (T[-1] - T_ambient) + epsilon_top * sigma * (T[-1]**4 - T_ambient**4)
            dTdt_transport[-1] = (flux[-1] - flux_out_top) / (rho_c_p_grid[-1] * (dx[-1]/2))
            dTdt = dTdt_source + dTdt_transport
            return dTdt
        
        # 솔버 실행
        sol = solve_ivp(fun=pde_system, t_span=[t_start, t_end], y0=T0, t_eval=t_eval, method='BDF')
        
        # Glass 부분의 x 좌표를 원래 크기로 복원
        x_restored = x.copy()
        if len(layer_indices_map) > 0 and len(thickness_layers_nm_original) > 0:
            glass_slice = layer_indices_map[0]
            glass_thickness_original = thickness_layers_nm_original[0] * 1e-9
            # Glass 부분의 x 좌표를 10000배 확장
            if glass_slice.stop <= len(x_restored) and glass_slice.start < glass_slice.stop:
                x_restored[glass_slice] = x[glass_slice] * glass_thickness_scale_factor
                # Glass 이후의 모든 좌표를 이동
                if glass_slice.stop > 0 and glass_slice.stop <= len(x):
                    glass_end_scaled = x[glass_slice.stop - 1]
                    if glass_slice.stop - 1 < len(x_restored):
                        glass_end_original = x_restored[glass_slice.stop - 1]
                        offset = glass_end_original - glass_end_scaled
                        if glass_slice.stop < len(x_restored):
                            x_restored[glass_slice.stop:] = x[glass_slice.stop:] + offset
        
        # Heat spreader와 Heat sink는 시각화 시 압축된 길이(100nm)로 표시하므로 복원하지 않음
        
        x_restored_nm = x_restored * 1e9
        
        # Glass와 ITO 경계점 찾기 및 활성층 부분 추출 (ITO 시작점을 x=0으로)
        glass_ito_boundary_idx = layer_indices_map[0].stop - 1
        glass_ito_boundary_nm = x_restored_nm[glass_ito_boundary_idx]
        active_start_idx = glass_ito_boundary_idx + 1
        
        # 활성층 위치 (ITO 시작점을 x=0으로 조정)
        position_active_nm = (x_restored_nm[active_start_idx:] - glass_ito_boundary_nm).tolist()
        
        # 활성층 온도
        temperature_active = sol.y[active_start_idx:, :].tolist()
        
        # Glass 부분 (축약 표시용)
        position_glass_nm = x_restored_nm[:active_start_idx].tolist()
        temperature_glass = sol.y[:active_start_idx, :].tolist()
        
        # 페로브스카이트 중간 지점에서의 시간에 따른 온도
        if perovskite_layer_index < len(layer_indices_map):
            perovskite_start_idx = layer_indices_map[perovskite_layer_index].start
            perovskite_end_idx = layer_indices_map[perovskite_layer_index].stop
            perovskite_mid_idx = (perovskite_start_idx + perovskite_end_idx) // 2
            if perovskite_mid_idx < sol.y.shape[0]:
                perovskite_center_temp = sol.y[perovskite_mid_idx, :].tolist()
            else:
                # 안전한 대체값: 첫 번째 노드의 온도 사용
                perovskite_center_temp = sol.y[0, :].tolist() if sol.y.shape[0] > 0 else []
        else:
            # perovskite_layer_index가 유효하지 않은 경우 첫 번째 노드 사용
            perovskite_center_temp = sol.y[0, :].tolist() if sol.y.shape[0] > 0 else []
        
        # 활성층 레이어 경계 (ITO 시작점을 x=0으로)
        active_layer_boundaries_nm = [0.0]  # ITO 시작점 (x=0)
        # Glass를 제외한 활성층 레이어들 (인덱스 1부터 시작)
        # layer_names와 thickness_layers_nm_original의 길이가 같아야 하지만, 안전을 위해 더 작은 길이 사용
        try:
            max_idx = min(len(layer_names), len(thickness_layers_nm_original))
            for i in range(1, max_idx):
                if i >= len(thickness_layers_nm_original):
                    raise IndexError(f"active_layer_boundaries_nm 계산 중 인덱스 오류: i={i}, thickness_layers_nm_original 길이={len(thickness_layers_nm_original)}, layer_names={layer_names}")
                active_layer_boundaries_nm.append(float(active_layer_boundaries_nm[-1] + thickness_layers_nm_original[i]))
        except (IndexError, ValueError) as e:
            raise ValueError(f"active_layer_boundaries_nm 계산 중 오류: {str(e)}, layer_names 길이={len(layer_names)}, thickness_layers_nm_original 길이={len(thickness_layers_nm_original)}, layer_names={layer_names}") from e
        
        # NumPy 타입을 Python 기본 타입으로 변환하는 헬퍼 함수
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
        
        # 결과 반환
        result = {
            'success': True,
            'time': convert_to_python_type(sol.t.tolist()),
            'position_active_nm': convert_to_python_type(position_active_nm),
            'temperature_active': convert_to_python_type(temperature_active),
            'position_glass_nm': convert_to_python_type(position_glass_nm),
            'temperature_glass': convert_to_python_type(temperature_glass),
            'perovskite_center_temp': convert_to_python_type(perovskite_center_temp),
            'layer_boundaries_nm': convert_to_python_type(active_layer_boundaries_nm),
            'layer_names': layer_names[1:] if len(layer_names) > 1 else [],  # Glass 제외
            'glass_ito_boundary_nm': float(glass_ito_boundary_nm)
        }
        
        return jsonify(result)
        
    except Exception as e:
        # 상세한 오류 정보 수집
        error_traceback = traceback.format_exc()
        error_info = {
            'error_type': type(e).__name__,
            'error_message': str(e),
            'traceback': error_traceback
        }
        error_result = {
            'success': False, 
            'error': str(e),
            'error_details': error_info
        }
        return jsonify(error_result), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)

