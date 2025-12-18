from http.server import BaseHTTPRequestHandler
import json
import numpy as np
from scipy.integrate import solve_ivp

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        return
    
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
            
            # 파라미터 추출
            layer_names = data['layer_names']
            k_therm_layers_original = np.array(data['k_therm_layers'])
            rho_layers = np.array(data['rho_layers'])
            c_p_layers = np.array(data['c_p_layers'])
            thickness_layers_nm_original = np.array(data['thickness_layers_nm'])
            
            # Glass 두께를 10000배 줄여서 계산량 감소
            glass_thickness_scale_factor = 10000.0
            thickness_layers_nm = thickness_layers_nm_original.copy()
            thickness_layers_nm[0] = thickness_layers_nm_original[0] / glass_thickness_scale_factor  # Glass만 10000배 얇게
            thickness_layers = thickness_layers_nm * 1e-9
            
            # Glass의 effective 물성 계산
            # 1. Thermal conductivity (열 저항 유지: R = L/k)
            k_therm_layers = k_therm_layers_original.copy()
            k_therm_layers[0] = k_therm_layers_original[0] * glass_thickness_scale_factor  # Glass만 10000배 증가
            
            # 2. Density와 Heat capacity (열용량 및 시간 상수 유지)
            rho_layers_effective = rho_layers.copy()
            c_p_layers_effective = c_p_layers.copy()
            rho_layers_effective[0] = rho_layers[0] * glass_thickness_scale_factor  # Glass만 10000배 증가
            
            voltage = data['voltage']
            current_density = data['current_density']
            Q_A = voltage * current_density
            
            epsilon_top = data['epsilon_top']
            epsilon_bottom = data['epsilon_bottom']
            sigma = 5.67e-8
            h_conv = data['h_conv']
            T_ambient = data['T_ambient']
            
            t_start = data.get('t_start', 0)
            t_end = data.get('t_end', 1000.0)
            t_eval = np.linspace(t_start, t_end, 200)
            
            # 비균일 그리드 및 물성 배열 생성
            points_per_layer = [50, 20, 20, 40, 20, 20]
            x_nodes = [0.0]
            layer_indices_map = []
            start_idx = 0
            for i, thickness in enumerate(thickness_layers):
                num_points = points_per_layer[i]
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
            for i, prop_slice in enumerate(layer_indices_map):
                k_grid[prop_slice] = k_therm_layers[i]
                rho_c_p_grid[prop_slice] = rho_layers_effective[i] * c_p_layers_effective[i]
            
            # 열원 위치 계산
            perovskite_layer_index = 3
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
            glass_slice = layer_indices_map[0]
            glass_thickness_original = thickness_layers_nm_original[0] * 1e-9
            # Glass 부분의 x 좌표를 10000배 확장
            x_restored[glass_slice] = x[glass_slice] * glass_thickness_scale_factor
            # Glass 이후의 모든 좌표를 이동
            glass_end_scaled = x[glass_slice.stop - 1]
            glass_end_original = x_restored[glass_slice.stop - 1]
            offset = glass_end_original - glass_end_scaled
            x_restored[glass_slice.stop:] = x[glass_slice.stop:] + offset
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
            perovskite_start_idx = layer_indices_map[3].start
            perovskite_end_idx = layer_indices_map[3].stop
            perovskite_mid_idx = (perovskite_start_idx + perovskite_end_idx) // 2
            perovskite_center_temp = sol.y[perovskite_mid_idx, :].tolist()
            
            # 활성층 레이어 경계 (ITO 시작점을 x=0으로)
            active_layer_boundaries_nm = [0.0]  # ITO 시작점 (x=0)
            for i in range(1, len(layer_names)):
                active_layer_boundaries_nm.append(float(active_layer_boundaries_nm[-1] + thickness_layers_nm_original[i]))
            
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
                'layer_names': layer_names[1:],  # Glass 제외
                'glass_ito_boundary_nm': float(glass_ito_boundary_nm)
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
            
        except Exception as e:
            error_result = {'success': False, 'error': str(e)}
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(error_result).encode('utf-8'))
    
    def do_GET(self):
        result = {'status': 'ok', 'message': 'Backend is running'}
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode('utf-8'))

    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
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
        thickness_layers_nm[0] = thickness_layers_nm_original[0] / glass_thickness_scale_factor  # Glass만 10000배 얇게
        thickness_layers = thickness_layers_nm * 1e-9
        
        # Glass의 effective 물성 계산
        # 1. Thermal conductivity (열 저항 유지: R = L/k)
        k_therm_layers = k_therm_layers_original.copy()
        k_therm_layers[0] = k_therm_layers_original[0] * glass_thickness_scale_factor  # Glass만 10000배 증가
        
        # 2. Density와 Heat capacity (열용량 및 시간 상수 유지)
        rho_layers_effective = rho_layers.copy()
        c_p_layers_effective = c_p_layers.copy()
        rho_layers_effective[0] = rho_layers[0] * glass_thickness_scale_factor  # Glass만 10000배 증가
        
        voltage = data['voltage']
        current_density = data['current_density']
        Q_A = voltage * current_density
        
        epsilon_top = data['epsilon_top']
        epsilon_bottom = data['epsilon_bottom']
        sigma = 5.67e-8
        h_conv = data['h_conv']
        T_ambient = data['T_ambient']
        
        t_start = data.get('t_start', 0)
        t_end = data.get('t_end', 1000.0)
        t_eval = np.linspace(t_start, t_end, 200)
        
        # 비균일 그리드 및 물성 배열 생성
        points_per_layer = [50, 20, 20, 40, 20, 20]
        x_nodes = [0.0]
        layer_indices_map = []
        start_idx = 0
        for i, thickness in enumerate(thickness_layers):
            num_points = points_per_layer[i]
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
        for i, prop_slice in enumerate(layer_indices_map):
            k_grid[prop_slice] = k_therm_layers[i]
            rho_c_p_grid[prop_slice] = rho_layers_effective[i] * c_p_layers_effective[i]
        
        # 열원 위치 계산
        perovskite_layer_index = 3
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
        glass_slice = layer_indices_map[0]
        glass_thickness_original = thickness_layers_nm_original[0] * 1e-9
        # Glass 부분의 x 좌표를 10000배 확장
        x_restored[glass_slice] = x[glass_slice] * glass_thickness_scale_factor
        # Glass 이후의 모든 좌표를 이동
        glass_end_scaled = x[glass_slice.stop - 1]
        glass_end_original = x_restored[glass_slice.stop - 1]
        offset = glass_end_original - glass_end_scaled
        x_restored[glass_slice.stop:] = x[glass_slice.stop:] + offset
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
        perovskite_start_idx = layer_indices_map[3].start
        perovskite_end_idx = layer_indices_map[3].stop
        perovskite_mid_idx = (perovskite_start_idx + perovskite_end_idx) // 2
        perovskite_center_temp = sol.y[perovskite_mid_idx, :].tolist()
        
        # 활성층 레이어 경계 (ITO 시작점을 x=0으로)
        active_layer_boundaries_nm = [0.0]  # ITO 시작점 (x=0)
        for i in range(1, len(layer_names)):
            active_layer_boundaries_nm.append(float(active_layer_boundaries_nm[-1] + thickness_layers_nm_original[i]))
        
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
            'layer_names': layer_names[1:],  # Glass 제외
            'glass_ito_boundary_nm': float(glass_ito_boundary_nm)
        }
        

