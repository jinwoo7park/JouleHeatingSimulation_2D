import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 1. 파라미터 설정
# 시간 설정
t_start = 0
t_end = 1000.0 # 긴 시간으로 안정 상태 확인
t_eval = np.linspace(t_start, t_end, 200)


layer_names = ['Glass', 'ITO', 'HTL', 'Perovskite', 'ETL', 'Cathode']
k_therm_layers_original = np.array([0.8, 10.0, 0.2, 0.5, 0.2, 200.0]) # Thermal conductivity (original)
rho_layers = np.array([2500, 7140, 1000, 4100, 1200, 2700]) # Density
c_p_layers = np.array([1000, 280, 1500, 250, 1500, 900]) # Thermal capacity
thickness_layers_nm_original = np.array([1.1e6, 70, 80, 280, 50, 100]) # Thickness nm (original)

# Glass 두께를 10000배 줄여서 계산량 감소
glass_thickness_scale_factor = 10000.0
thickness_layers_nm = thickness_layers_nm_original.copy()
thickness_layers_nm[0] = thickness_layers_nm_original[0] / glass_thickness_scale_factor  # Glass만 10000배 얇게
thickness_layers = thickness_layers_nm * 1e-9

# Glass의 effective 물성 계산
# 1. Thermal conductivity (열 저항 유지: R = L/k)
# 원래 열 저항: R_original = L_original / k_original
# 새로운 열 저항: R_new = L_new / k_new = (L_original/10000) / k_new
# R_original = R_new 이므로: L_original / k_original = (L_original/10000) / k_new
# 따라서: k_new = k_original * 10000
k_therm_layers = k_therm_layers_original.copy()
k_therm_layers[0] = k_therm_layers_original[0] * glass_thickness_scale_factor  # Glass만 10000배 증가

# 2. Density와 Heat capacity (열용량 및 시간 상수 유지)
# 열용량 = ρ·cp·L (단위 면적당)
# L이 10000배 줄어드므로, ρ·cp를 10000배 증가시켜야 전체 열용량이 동일
# 시간 상수: τ ~ L²·ρ·cp/k
# L을 10000배 줄이고, k를 10000배 증가시키면, ρ·cp도 10000배 증가시켜야 τ가 동일
rho_layers_effective = rho_layers.copy()
c_p_layers_effective = c_p_layers.copy()
rho_layers_effective[0] = rho_layers[0] * glass_thickness_scale_factor  # Glass만 10000배 증가
# 또는 c_p만 증가시켜도 됨: c_p_layers_effective[0] = c_p_layers[0] * glass_thickness_scale_factor

voltage = 2.9 # Operating Voltage (V)
current_density = 300.0 # Current density (A/m^2, = 10* mA/cm^2)
Q_A = voltage * current_density

epsilon_top = 0.05 # Emissivity_top (cathode)
epsilon_bottom = 0.85 # Emissivity)_bottom (glass)
sigma = 5.67e-8
h_conv = 10.0 # Natural convection constant
T_ambient = 25.0 + 273.15  # Environmental temperature in Kelvin (25°C)

# 2. 비균일 그리드 및 물성 배열 생성
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

# 3. 열원 위치 계산
perovskite_layer_index = 3
perovskite_slice = layer_indices_map[perovskite_layer_index]
L_perovskite = thickness_layers[perovskite_layer_index]
C_source_term = Q_A / (L_perovskite * rho_layers[perovskite_layer_index] * c_p_layers[perovskite_layer_index])

T0 = np.full(Nx, T_ambient)

# 4. PDE 시스템 정의
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

# 5. 솔버 실행 및 시각화
print("최종 모델로 시뮬레이션 중...")
sol = solve_ivp(fun=pde_system, t_span=[t_start, t_end], y0=T0, t_eval=t_eval, method='BDF')
print("계산 완료.")

# Glass 부분의 x 좌표를 원래 크기로 복원
x_restored = x.copy()
glass_slice = layer_indices_map[0]
glass_thickness_original = thickness_layers_nm_original[0] * 1e-9
glass_thickness_scaled = thickness_layers[0]
# Glass 부분의 x 좌표를 10000배 확장
x_restored[glass_slice] = x[glass_slice] * glass_thickness_scale_factor
# Glass 이후의 모든 좌표를 이동 (Glass 끝점이 원래 위치로 이동했으므로 offset 계산)
glass_end_scaled = x[glass_slice.stop - 1]  # 축소된 버전의 Glass 끝점
glass_end_original = x_restored[glass_slice.stop - 1]  # 복원된 Glass 끝점
offset = glass_end_original - glass_end_scaled  # Glass 끝점의 이동량
x_restored[glass_slice.stop:] = x[glass_slice.stop:] + offset

# --- 시각화 1: 최종 온도 프로파일 (Glass를 물결선으로 축약, ITO 시작점을 x=0으로) ---
time_index = -1
actual_time = sol.t[time_index]
T_profile = sol.y[:, time_index]
# 켈빈을 섭씨로 변환
T_profile_C = T_profile - 273.15
x_nm = x_restored * 1e9  # 복원된 x 좌표 사용

# Glass와 ITO 경계점 찾기
glass_ito_boundary_idx = layer_indices_map[0].stop - 1
glass_ito_boundary_nm = x_nm[glass_ito_boundary_idx]

# 활성층 부분만 추출 (ITO부터)
active_start_idx = glass_ito_boundary_idx + 1
x_active_nm = x_nm[active_start_idx:] - glass_ito_boundary_nm  # ITO 시작점을 x=0으로 조정
T_active = T_profile_C[active_start_idx:]

# Glass 부분 추출 (축약 표시용)
x_glass_nm = x_nm[:active_start_idx]
T_glass = T_profile_C[:active_start_idx]

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle(f'1D Temperature Profile at t = {actual_time:.1f} s', fontsize=16)

# 활성층 부분 그리기 (실선)
ax.plot(x_active_nm, T_active, '-', linewidth=2, label='Active layers', color='blue')

# Glass 부분을 물결선으로 축약 표시
# 물결 모양의 곡선 생성 (x=-200 ~ 0 nm)
n_points = 100
wavy_x = np.linspace(-200, 0, n_points)
# Glass의 시작과 끝 온도를 선형 보간
glass_start_temp = T_glass[0]
glass_end_temp = T_glass[-1]
T_base = np.linspace(glass_start_temp, glass_end_temp, n_points)
# 물결 모양 추가 (작은 진폭)
amplitude = (glass_end_temp - glass_start_temp) * 0.02  # 온도 차이의 2% 정도
wavy_pattern = amplitude * np.sin(np.linspace(0, 4 * np.pi, n_points))
wavy_T = T_base + wavy_pattern
ax.plot(wavy_x, wavy_T, '-', linewidth=2, label='Glass (compressed)', color='red', alpha=0.7)

ax.set_xlabel('Position from ITO/Glass interface (nm)', fontsize=12)
ax.set_ylabel('Temperature (°C)', fontsize=12)
ax.grid(True, linestyle=':', alpha=0.7)
ax.legend()

# 활성층 레이어 경계 및 레이블 추가
current_pos_nm = 0
layer_boundary_nm_list = [0]  # ITO 시작점 (x=0)
for i in range(1, len(layer_names)):  # Glass 제외
    current_pos_nm += thickness_layers_nm_original[i]
    layer_boundary_nm_list.append(current_pos_nm)
    ax.axvline(x=current_pos_nm, color='gray', linestyle='--', alpha=0.7)
    layer_center_nm = (layer_boundary_nm_list[-2] + current_pos_nm) / 2
    ax.text(layer_center_nm, ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
            layer_names[i], ha='center', va='bottom', fontsize=10, rotation=90)

ax.set_xlim(-250, np.max(x_active_nm) * 1.05)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# --- 시각화 2: 페로브스카이트 중간 지점에서의 시간에 따른 온도 ---
# 페로브스카이트 레이어의 중간 인덱스 찾기
perovskite_start_idx = layer_indices_map[perovskite_layer_index].start
perovskite_end_idx = layer_indices_map[perovskite_layer_index].stop
perovskite_mid_idx = (perovskite_start_idx + perovskite_end_idx) // 2
T_perovskite_mid = sol.y[perovskite_mid_idx, :]  # 시간에 따른 온도 (켈빈)
T_perovskite_mid_C = T_perovskite_mid - 273.15  # 섭씨로 변환

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(sol.t, T_perovskite_mid_C, '-', linewidth=2, color='green', label='Perovskite center temperature')
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('Temperature (°C)', fontsize=12)
ax2.set_title('Temperature at Perovskite Layer Center vs Time', fontsize=14)
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend()
plt.tight_layout()
plt.show()