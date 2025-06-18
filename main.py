import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ====================== 1. 数据加载与预处理 ======================
df = pd.read_csv("B 任务数据集.csv")
time = df['time'].values
u = df['volte'].values
y = df['temperature'].values

# 去除初始稳态段（假设前50点为稳态）
y_initial = np.mean(y[:50])
step_idx = np.argmax(u > u[0] * 1.1)  # 检测阶跃起始点
t_step = time[step_idx]
y_step = y[step_idx:]
t_step = time[step_idx:] - t_step  # 以阶跃时刻为时间起点


# ====================== 2. 系统辨识（带延迟的一阶模型） ======================
def identify_first_order_with_delay(t, y, y_initial, threshold=0.05):
    """辨识带延迟的一阶模型：G(s) = K e^(-Ls)/(Ts + 1)"""
    #计算比例系数 K
    delta_y = y[-1] - y_initial  # 稳态输出变化
    K = delta_y / (u.max() - 0)  # 增益（假设输入为阶跃信号）

    # 计算延迟时间 L（达到threshold*delta_y的时间）
    target = y_initial + threshold * delta_y
    L_mask = y > target
    if np.any(L_mask):
        L = t[L_mask][0]
    else:
        L = 0  # 无明显延迟

    # 计算时间常数 T（去除延迟后的63%响应时间）
    y_adj = y - y_initial
    y_adj = y_adj[L_mask] if L > 0 else y_adj
    t_adj = t[L_mask] if L > 0 else t
    if len(y_adj) == 0:
        T = 0
    else:
        target_T = 0.63 * delta_y
        T_mask = y_adj > target_T
        if np.any(T_mask):
            T = t_adj[T_mask][0] - L
        else:
            T = np.inf  # 未达到63%响应

    return K, L, T


K, L, T = identify_first_order_with_delay(t_step, y_step, y_initial, threshold=0.05)
print(f"辨识结果：K={K:.4f}, L={L:.2f}s, T={T:.2f}s")



#-----------------------------------------------------------------------------------------------------------------------
#2.验证辨识模型
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


# 重新构建时间轴，仿真和原始数据对齐
t_sim = time.copy()

# 构建传递函数 G(s) = K / (T*s + 1)
num = [K]
den = [T, 1]

# 构建离散时间序列仿真信号
system = signal.TransferFunction(num, den)

# 用 step response 进行仿真（单位阶跃输入 1V），咱们手动乘 delta_u 来对齐实际阶跃
t_out, y_out_no_delay = signal.step(system, T=t_sim)

# 加入阶跃幅值 + 初始温度
delta_u = u.max()-0
y_out_no_delay = y_initial + delta_u * y_out_no_delay

# 人为加延迟：L_time 对应多少个样本点
dt = t_sim[1] - t_sim[0]  # 采样周期
delay_steps = int(L / dt)

# 构造延迟后的信号
y_out = np.concatenate([np.ones(delay_steps) * y_initial, y_out_no_delay[:-delay_steps]])

# 画对比图
plt.figure(figsize=(10, 6))
plt.plot(time, y, label='Original Temperature (Measured)', color='tab:red')
plt.plot(t_sim, y_out, label='Identified Model Response', linestyle='--', color='tab:blue')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.title('Model Verification: Identified Model vs. Real Data')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#-----------------------------------------------------------------------------------------------------------------------
#3.智能优化算法PSO优化PID参数
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from pyswarm import pso  # 粒子群优化算法库，pip install pyswarm

# === 固定部分 ===
# 你的传递函数参数
K_process = K
L_process = L
T_process = T

# 仿真时间
t_sim = np.linspace(0, 10000, 10000)  # 10000秒, 1s 采样

# 设定值 35℃
setpoint = 35.0


# === PID 控制器仿真 ===
def pid_sim(Kp, Ki, Kd):
    # PID 控制器传递函数
    num_pid = [Kd, Kp, Ki]
    den_pid = [1, 0]

    # 被控对象传递函数 G(s)
    num_process = [K_process]
    den_process = [T_process, 1]

    # 手动串联 PID * Process
    num_open = np.polymul(num_pid, num_process)
    den_open = np.polymul(den_pid, den_process)

    # 手动构造闭环系统 H_cl = H_open / (1 + H_open)
    num_closed = num_open
    den_closed = np.polyadd(den_open, num_open)

    closed_loop = signal.TransferFunction(num_closed, den_closed)

    # 仿真闭环 step response
    t_out, y_out_no_delay = signal.step(closed_loop, T=t_sim)

    # 加延迟
    dt = t_sim[1] - t_sim[0]
    delay_steps = int(L_process / dt)
    y_out = np.concatenate([np.ones(delay_steps) * y_out_no_delay[0], y_out_no_delay[:-delay_steps]])

    # 乘 setpoint
    y_out *= setpoint

    return t_out, y_out

# === 性能指标 ===
def performance_index(params):
    Kp, Ki, Kd = params
    t_out, y_out = pid_sim(Kp, Ki, Kd)

    # 计算 ISE
    error = y_out - setpoint
    ISE = np.sum(error ** 2)

    return ISE


# === 参数优化 ===
# PID 参数范围 [Kp_min, Kp_max], [Ki_min, Ki_max], [Kd_min, Kd_max]
lb = [0, 0, 0]
ub = [10.0, 1.0, 5.0]

# 用粒子群优化PSO自动搜索 PID 最优参数
best_params, best_score = pso(performance_index, lb, ub, swarmsize=20, maxiter=30)

# === 最优 PID 参数仿真 ===
Kp_opt, Ki_opt, Kd_opt = best_params
t_out, y_out = pid_sim(Kp_opt, Ki_opt, Kd_opt)
print("最优PID参数：",Kp_opt,Ki_opt,Kd_opt)

# === 画图 ===
plt.figure(figsize=(10, 6))
plt.plot(t_out, y_out, label='Optimized PID Response', color='tab:blue')
plt.axhline(setpoint, color='tab:green', linestyle='--', label='Setpoint = 35℃')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.title(f'Optimized PID Response\nKp={Kp_opt:.2f}, Ki={Ki_opt:.2f}, Kd={Kd_opt:.2f}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#-----------------------------------------------------------------------------------------------------------------------
#4.性能指标计算模板
def compute_performance_metrics(t_out, y_out, setpoint, tolerance=0.02):
    # 上升时间（5% 到 95%）
    y_5 = setpoint * 0.05
    y_95 = setpoint * 0.95

    try:
        t_rise_start = t_out[np.where(y_out >= y_5)[0][0]]
        t_rise_end = t_out[np.where(y_out >= y_95)[0][0]]
        rise_time = t_rise_end - t_rise_start
    except:
        rise_time = np.nan  # 如果没达到 95%，就 NaN

    # 超调量
    max_temp = np.max(y_out)
    overshoot = ((max_temp - setpoint) / setpoint) * 100

    # 稳态误差
    y_steady_state = np.mean(y_out[-100:])  # 最后100个样本点
    steady_state_error = abs(setpoint - y_steady_state)

    # 调整时间
    lower_bound = setpoint * (1 - tolerance)
    upper_bound = setpoint * (1 + tolerance)

    settling_time = np.nan  # 默认NaN
    for i in range(len(y_out)):
        if np.all((y_out[i:] >= lower_bound) & (y_out[i:] <= upper_bound)):
            settling_time = t_out[i]
            break

    # 返回结果
    metrics = {
        "Rise Time (s)": rise_time,
        "Overshoot (%)": overshoot,
        "Steady-State Error (°C)": steady_state_error,
        "Settling Time (s)": settling_time
    }

    return metrics



metrics = compute_performance_metrics(t_out, y_out, setpoint)

# 打印性能指标
for k, v in metrics.items():
    print(f"{k}: {v:.2f}")