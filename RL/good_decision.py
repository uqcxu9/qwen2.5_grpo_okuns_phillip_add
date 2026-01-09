import pickle as pkl
import numpy as np
import pandas as pd
import json
import sys

# 导入 reward
sys.path.append(r'/workspace/QWEN2.5_42_7b_main/RL')
from reward import compute_score

data_path = r'/workspace/QWEN2.5_42_7b_main/data/gpt-3-noperception-reflection-1-100agents-240months'

with open(f'{data_path}/dense_log.pkl', 'rb') as f:
    dense_log = pkl.load(f)

# 从dense_log中提取三个关键数据：状态、行动、税收
states = dense_log['states']
actions = dense_log['actions']
periodic_tax = dense_log['PeriodicTax']

print(f"Total timesteps: {len(states)}")
print(f"Length of actions: {len(actions)}")
print(f"Length of periodic_tax: {len(periodic_tax)}")

# 动态获取 agent IDs,排除planner
agent_ids = sorted([aid for aid in states[0].keys() if aid != "p" and isinstance(states[0][aid], dict)])
num_agents = len(agent_ids)
print(f"Number of agents: {num_agents}")

# 2. 加载价格数据
class DummyUnpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if 'ai_economist' in module:
            return type(name, (), {})
        return super().find_class(module, name)

env_file = f'{data_path}/env_240.pkl'
with open(env_file, "rb") as f:
    env = DummyUnpickler(f).load()

prices = list(env.world.price)
print(f"Length of price data: {len(prices)}")

# 真实数据
real_monthly_unemployment = [
    4.2,4.2,4.3,4.4,4.3,4.5,4.6,4.9,5.0,5.3,5.5,5.7,  # 2001
    5.7,5.7,5.7,5.9,5.8,5.8,5.8,5.7,5.7,5.7,5.9,6.0,  # 2002
    5.8,5.9,5.9,6.0,6.1,6.3,6.2,6.1,6.1,6.0,5.8,5.7,  # 2003
    5.7,5.6,5.8,5.6,5.6,5.6,5.5,5.4,5.4,5.5,5.4,5.4,  # 2004
    5.3,5.4,5.2,5.2,5.1,5.0,5.0,4.9,5.0,5.0,5.0,4.9,  # 2005
    4.7,4.8,4.7,4.7,4.6,4.6,4.7,4.7,4.5,4.4,4.5,4.4,  # 2006
    4.6,4.5,4.4,4.5,4.4,4.6,4.7,4.6,4.7,4.7,4.7,5.0,  # 2007
    5.0,4.9,5.1,5.0,5.4,5.6,5.8,6.1,6.1,6.5,6.8,7.3,  # 2008
    7.8,8.3,8.7,9.0,9.4,9.5,9.5,9.6,9.8,10.0,9.9,9.9, # 2009
    9.8,9.8,9.9,9.9,9.6,9.4,9.4,9.5,9.5,9.4,9.8,9.3,  # 2010
    9.1,9.0,9.0,9.1,9.0,9.1,9.0,9.0,9.0,8.8,8.6,8.5,  # 2011
    8.3,8.3,8.2,8.2,8.2,8.2,8.2,8.1,7.8,7.8,7.7,7.9,  # 2012
    8.0,7.7,7.5,7.6,7.5,7.5,7.3,7.2,7.2,7.2,6.9,6.7,  # 2013
    6.6,6.7,6.7,6.2,6.3,6.1,6.2,6.1,5.9,5.7,5.8,5.6,  # 2014
    5.7,5.5,5.4,5.4,5.6,5.3,5.2,5.1,5.0,5.0,5.1,5.0,  # 2015
    4.8,4.9,5.0,5.1,4.8,4.9,4.8,4.9,5.0,4.9,4.7,4.7,  # 2016
    4.7,4.6,4.4,4.4,4.4,4.3,4.3,4.4,4.3,4.2,4.2,4.1,  # 2017
    4.0,4.1,4.0,4.0,3.8,4.0,3.8,3.8,3.7,3.8,3.8,3.9,  # 2018
    4.0,3.8,3.8,3.7,3.6,3.6,3.7,3.6,3.5,3.6,3.6,3.6,  # 2019
    3.6,3.5,4.4,14.8,13.2,11.0,10.2,8.4,7.8,6.9,6.7,6.7, # 2020
    6.4,6.2,6.1,6.1,5.8,5.9,5.4,5.1,4.7,4.5,4.2,3.9   # 2021
]

real_wage_inflation = [
    3.675, 3.2, 2.9, 2.625, 2.5, 2.85, 3.4, 2.95, 1.575,  # 2001-2009
    1.625, 1.65, 1.8, 1.875, 2.025, 2.3, 2.325, 2.6, 3.0, 3.0, 2.925, 4.025  # 2010-2021
]

real_gdp_growth = [
    1.70, 2.80, 3.85, 3.48, 2.78, 2.00, 0.11, -2.58,  # 2002-2009
    2.70, 1.56, 2.29, 2.12, 2.52, 2.95, 1.82, 2.46, 2.97, 2.58, -2.16  # 2010-2020
]

# 计算年度平均失业率
real_yearly_unemp = []
for year_idx in range(21):
    start = year_idx * 12
    end = start + 12
    real_yearly_unemp.append(np.mean(real_monthly_unemployment[start:end]))

# 计算失业率变化 (2002-2021，共20个值)
real_unemp_change = []
for i in range(1, len(real_yearly_unemp)):
    real_unemp_change.append(real_yearly_unemp[i] - real_yearly_unemp[i-1])

# 计算经济规律系数
okun_gdp = real_gdp_growth  
okun_unemp = real_unemp_change[:19] 

# 线性回归: unemp_change = α + β * gdp_growth
okun_beta = -1.033
okun_alpha = np.mean(real_unemp_change[:19]) - okun_beta * np.mean(real_gdp_growth)

# 计算残差（实际值 - 预测值）
# 计算残差的标准差和90%分位数（用作阈值）
okun_residuals = [okun_unemp[i] - (okun_alpha + okun_beta * okun_gdp[i]) for i in range(len(okun_gdp))]
okun_residual_std = np.std(okun_residuals)
okun_tau = np.quantile(np.abs(okun_residuals), 0.90)

print(f"Okun's Law: ΔUnemp = {okun_alpha:.3f} + {okun_beta:.3f} × RealOutputGrowth")
print(f"Okun 残差标准差: {okun_residual_std:.3f}pp, P90阈值: {okun_tau:.3f}pp")


phillips_unemp = real_yearly_unemp  
phillips_wage = real_wage_inflation  

phillips_beta = -0.246
phillips_alpha = np.mean(real_wage_inflation) - phillips_beta * np.mean(real_yearly_unemp)

phillips_residuals = [phillips_wage[i] - (phillips_alpha + phillips_beta * phillips_unemp[i]) for i in range(len(phillips_unemp))]
phillips_residual_std = np.std(phillips_residuals)
phillips_tau = np.quantile(np.abs(phillips_residuals), 0.90)

print(f"Okun's Law (from eval script): ΔUnemp = {okun_alpha:.3f} + {okun_beta:.3f} × RealOutputGrowth")
print(f"Phillips Curve (from eval script): WageInfl = {phillips_alpha:.3f} + {phillips_beta:.3f} × Unemp")

#  Helper functions 
A = 1  # productivity
num_labor_hours = 168 
def get_work_from_state(state, num_labor_hours=168):
    """work ∈ [0,1] from endogenous['Labor'] (0 or 168)"""
    labor_hours = state.get("endogenous", {}).get("Labor", None)
    if labor_hours is None:
        return None
    try:
        return float(np.clip(float(labor_hours) / float(num_labor_hours), 0.0, 1.0))
    except:
        return None

# 计算DPI
def calculate_dpi(t, agent_id_str, states, periodic_tax):
    """Calculate DPI = income + lump_sum - tax_paid"""
    income = states[t][agent_id_str]['income']['Coin']
    lump_sum = periodic_tax[t].get(agent_id_str, {}).get('lump_sum', 0)
    tax_paid = periodic_tax[t].get(agent_id_str, {}).get('tax_paid', 0)
    return income + lump_sum - tax_paid

# 计算月度实际产出
def calculate_monthly_gdp(t, states, agent_ids):
    """计算月度 GDP（基于 income，与 reward.py 一致）"""
    total = 0.0
    for agent_id_str in agent_ids:
        agent_state = states[t].get(agent_id_str, {})
        income = agent_state.get('income', {})
        if isinstance(income, dict):
            total += income.get('Coin', 0)
        else:
            total += float(income) if income else 0
    return total

# 计算年度平均失业率
def calculate_yearly_unemployment(year, states, max_t):
    year_start = (year - 1) * 12
    year_end = year * 12
    if year_end > max_t:
        return None
    
    monthly_unemp = []
    for t in range(year_start, year_end):
        unemployed = 0
        total = 0
        for aid, state in states[t].items():
            if aid == "p" or not isinstance(state, dict):
                continue
            total += 1
            job = state.get("endogenous", {}).get("job")
            if job == "Unemployment":
                unemployed += 1
        if total > 0:
            monthly_unemp.append(unemployed / total)
    
    return np.mean(monthly_unemp) if monthly_unemp else None

# 获取所有月份的失业率，用于计算分位数
def get_monthly_unemployment_rates(states, max_t):
    monthly_rates = []
    for t in range(max_t):
        unemployed = 0
        total = 0
        for aid, state in states[t].items():
            if aid == "p" or not isinstance(state, dict):
                continue
            total += 1
            job = state.get("endogenous", {}).get("job")
            if job == "Unemployment":
                unemployed += 1
        if total > 0:
            monthly_rates.append(unemployed / total)
        else:
            monthly_rates.append(None)
    return monthly_rates

# 计算年度wage通胀率（基于 skill）
def calculate_yearly_wage_inflation(year, states, max_t):
    year_start = (year - 1) * 12
    year_end = year * 12
    prev_year_start = (year - 2) * 12
    prev_year_end = (year - 1) * 12
    
    if year_end > max_t or prev_year_start < 0:
        return None
    
    def avg_skill(start, end):
        skills = []
        for t in range(start, end):
            for aid, state in states[t].items():
                if aid == "p" or not isinstance(state, dict):
                    continue
                s = state.get("skill")
                if s is not None:
                    skills.append(float(s))
        return np.mean(skills) if skills else None
    
    curr_skill = avg_skill(year_start, year_end)
    prev_skill = avg_skill(prev_year_start, prev_year_end)
    
    if curr_skill is None or prev_skill is None or prev_skill < 1e-6:
        return None
    
    return (curr_skill - prev_skill) / prev_skill * 100

# 3. 计算所有年度的宏观指标 

max_t = min(len(states), len(periodic_tax), len(actions), len(prices))
year_metrics = {}

for year in range(2, 21):
    year_start = (year - 1) * 12
    year_end = year * 12
    prev_year_start = (year - 2) * 12
    
    if year_end > max_t:
        break
    
    # 实际产出增长率（用于 Okun）
    curr_gdp = sum(calculate_monthly_gdp(t, states, agent_ids) for t in range(year_start, year_end))
    prev_gdp = sum(calculate_monthly_gdp(t, states, agent_ids) for t in range(prev_year_start, year_start))
    output_growth = (curr_gdp - prev_gdp) / prev_gdp * 100 if prev_gdp > 0 else None
    
    # price通胀率（用于范围检查）
    curr_price = np.mean([prices[t] for t in range(year_start, year_end)])
    prev_price = np.mean([prices[t] for t in range(prev_year_start, year_start)])
    price_inflation = (curr_price - prev_price) / prev_price * 100 if prev_price > 0 else None
    
    # 失业率
    curr_unemp = calculate_yearly_unemployment(year, states, max_t)
    prev_unemp = calculate_yearly_unemployment(year - 1, states, max_t)
    
    if curr_unemp is not None and prev_unemp is not None:
        unemp_change = (curr_unemp - prev_unemp) * 100  
    else:
        unemp_change = None
    
    wage_inflation = calculate_yearly_wage_inflation(year, states, max_t)

    year_metrics[year] = {
        'output_growth': output_growth,
        'price_inflation': price_inflation,
        'wage_inflation': wage_inflation,  # 新增
        'unemployment': curr_unemp,
        'unemp_change': unemp_change,
    }
    
    if all(v is not None for v in [output_growth, curr_unemp, unemp_change, price_inflation, wage_inflation]):
        print(f"Year {year}: Output={output_growth:.2f}%, Unemp={curr_unemp*100:.1f}%, "
              f"ΔUnemp={unemp_change:.2f}pp, PriceInfl={price_inflation:.2f}%, WageInfl={wage_inflation:.2f}%")
    else:
        print(f"Year {year}: Missing data")
def compute_trailing_12m_correlations(states, t, num_agents):
    """计算 trailing 12 个月的 Okun/Phillips 相关系数"""
    if t < 12:
        return None, None, None
    
    okun_pairs = []
    phil_pairs = []
    u_values = []
    
    for tt in range(t - 12, t):
        u = get_monthly_unemployment_rate_at_t(states, tt, num_agents)
        du = compute_unemp_change_at_t(states, tt, num_agents)
        g = compute_gdp_growth_at_t(states, tt, num_agents)
        w_infl = compute_wage_inflation_at_t(states, tt, num_agents)
        
        if du is not None and g is not None:
            okun_pairs.append((du, g))
        if u is not None and w_infl is not None:
            phil_pairs.append((u, w_infl))
        if u is not None:
            u_values.append(u)
    
    def safe_corr_pairs(pairs, min_n=6):
        if len(pairs) < min_n:
            return None
        xs, ys = zip(*pairs)
        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)
        if np.std(xs) < 1e-8 or np.std(ys) < 1e-8:
            return None
        return float(np.corrcoef(xs, ys)[0, 1])
    
    r_okun = safe_corr_pairs(okun_pairs)
    r_phil = safe_corr_pairs(phil_pairs)
    u_mean = float(np.mean(u_values)) if len(u_values) >= 6 else None
    
    return r_okun, r_phil, u_mean


def get_monthly_unemployment_rate_at_t(states, t, num_agents):
    """获取 t 时刻的失业率"""
    if t < 0 or t >= len(states):
        return None
    unemployed = 0
    valid_count = 0
    for agent_id_str in agent_ids:  # 改用 agent_ids
        agent_state = states[t].get(agent_id_str)
        if agent_state is None:
            continue
        valid_count += 1
        job_status = agent_state.get("endogenous", {}).get("job", None)
        if job_status == "Unemployment":
            unemployed += 1
    if valid_count == 0:
        return None
    return unemployed / valid_count


def compute_unemp_change_at_t(states, t, num_agents):
    """计算 t 时刻的失业率 YoY 变化（百分点）"""
    if t < 12:
        return None
    u_now = get_monthly_unemployment_rate_at_t(states, t, num_agents)
    u_12m = get_monthly_unemployment_rate_at_t(states, t - 12, num_agents)
    if u_now is None or u_12m is None:
        return None
    return (u_now - u_12m) * 100.0


def compute_gdp_growth_at_t(states, t, num_agents):
    """计算 t 时刻的 GDP YoY 增长率（百分比）"""
    if t < 12:
        return None
    
    def get_total_income(t_idx):
        if t_idx < 0 or t_idx >= len(states):
            return None
        total = 0.0
        for agent_id_str in agent_ids:  # 改用 agent_ids
            agent_state = states[t_idx].get(agent_id_str, {})
            income = agent_state.get('income', {})
            if isinstance(income, dict):
                total += income.get('Coin', 0)
            else:
                total += float(income) if income else 0
        return total

    gdp_now = get_total_income(t)
    gdp_12m_ago = get_total_income(t - 12)
    
    if gdp_now is None or gdp_12m_ago is None or gdp_12m_ago < 1e-6:
        return None
    return (gdp_now - gdp_12m_ago) / gdp_12m_ago * 100


def compute_wage_inflation_at_t(states, t, num_agents):
    """计算 t 时刻的工资通胀 YoY（百分比）"""
    if t < 12:
        return None
    
    def avg_skill(t_idx):
        vals = []
        for agent_id_str in agent_ids:  # 改用 agent_ids
            s = states[t_idx].get(agent_id_str)
            if s is None:
                continue
            v = s.get("skill", None)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except:
                continue
        if len(vals) == 0:
            return None
        return float(np.mean(vals))
    
    w_now = avg_skill(t)
    w_12m = avg_skill(t - 12)
    if w_now is None or w_12m is None or w_12m < 1e-8:
        return None
    return (w_now - w_12m) / w_12m * 100.0


def compute_price_inflation_at_t(prices, t):
    """计算 t 时刻的价格通胀 YoY（百分比）"""
    if t < 12 or t >= len(prices):
        return None
    p_now = prices[t]
    p_12m_ago = prices[t - 12]
    if p_now is None or p_12m_ago is None or p_12m_ago < 1e-6:
        return None
    return (p_now - p_12m_ago) / p_12m_ago * 100


def build_extra_info_for_reward(t, agent_id_str, states, actions, periodic_tax, prices, num_agents):
    """构建 reward_function 需要的 extra_info"""
    agent_id = int(agent_id_str)
    state = states[t].get(agent_id_str, {})
    
    income = state.get('income', {}).get('Coin', 0) if isinstance(state.get('income'), dict) else 0
    wealth = state.get('inventory', {}).get('Coin', 0) if isinstance(state.get('inventory'), dict) else 0
    skill = state.get('skill', 1.0)
    
    tax_info = periodic_tax[t].get(agent_id_str, {})
    tax_paid = tax_info.get('tax_paid', 0)
    lump_sum = tax_info.get('lump_sum', 0)
    
    dpi = income + lump_sum - tax_paid
    dpi_amt = max(dpi, 0.0)
    cash_on_hand = wealth + dpi_amt
    buffer_ratio = cash_on_hand / (dpi_amt + 1e-8) if dpi_amt > 1e-6 else 1.0
    buffer_ratio = max(0.0, min(10.0, buffer_ratio))
    
    # 统一使用 t_macro = t-1 避免泄漏
    t_macro = max(0, t - 1)
    unemp = get_monthly_unemployment_rate_at_t(states, t_macro, num_agents)
    gdp_g = compute_gdp_growth_at_t(states, t_macro, num_agents)
    unemp_change = compute_unemp_change_at_t(states, t_macro, num_agents)
    wage_infl = compute_wage_inflation_at_t(states, t_macro, num_agents)
    price_infl = compute_price_inflation_at_t(prices, t_macro)
    
    # 相关系数也用 t_macro（改动点）
    r_okun_12m, r_phil_12m, u_mean_12m = compute_trailing_12m_correlations(states, t_macro, num_agents)
    
    # 就业状态
    job = state.get('endogenous', {}).get('job')
    gt_employed = 0 if job == "Unemployment" else 1
    
    return {
        "timestep": t,
        "agent_id": agent_id,
        "income": float(income),
        "wealth": float(wealth),
        "dpi": float(dpi),
        "buffer_ratio": float(buffer_ratio),
        "tax_paid": float(tax_paid),
        "lump_sum": float(lump_sum),
        "skill": float(skill),
        "unemployment_rate": unemp,
        "gdp_growth": gdp_g,
        "unemp_change": unemp_change,
        "price_inflation": price_infl,
        "wage_inflation": wage_infl,
        "regime": None,
        "regime_strength": None,
        "r_okun_12m": r_okun_12m,
        "r_phil_12m": r_phil_12m,
        "u_mean_12m": u_mean_12m,
        "gt_employed": gt_employed,
        "job_status": job,
    }

#  计算仿真数据的 Okun/Phillips 残差分位数
sim_okun_res = []
sim_phillips_res = []
# 计算仿真数据相对于Okun定律和Phillips曲线的残差
for y, m in year_metrics.items():
    if m['output_growth'] is not None and m['unemp_change'] is not None:
        res = abs(m['unemp_change'] - (okun_alpha + okun_beta * m['output_growth']))
        sim_okun_res.append(res)
    
    if m['unemployment'] is not None and m['wage_inflation'] is not None:
        u_pct = m['unemployment'] * 100
        res = abs(m['wage_inflation'] - (phillips_alpha + phillips_beta * u_pct))
        sim_phillips_res.append(res)
#  取残差的90%分位数作为阈值（加保护）
if len(sim_okun_res) > 0:
    okun_tau_sim = np.quantile(sim_okun_res, 0.90)
else:
    okun_tau_sim = okun_tau  # 回退用真实数据的阈值
    print("[WARNING] sim_okun_res 为空，使用真实数据阈值")

if len(sim_phillips_res) > 0:
    phillips_tau_sim = np.quantile(sim_phillips_res, 0.90)
else:
    phillips_tau_sim = phillips_tau  # 回退用真实数据的阈值
    print("[WARNING] sim_phillips_res 为空，使用真实数据阈值")

print(f"仿真 Okun 残差: P50={np.median(sim_okun_res) if sim_okun_res else 0:.2f}pp, P90={okun_tau_sim:.2f}pp")
print(f"仿真 Phillips 残差: P50={np.median(sim_phillips_res) if sim_phillips_res else 0:.2f}%, P90={phillips_tau_sim:.2f}%")

# Okun/Phillips 一致性检查
# 检查：失业率变化是否符合Okun定律预测
def check_okun_consistency(output_growth, unemp_change):
    if output_growth is None or unemp_change is None:
        return True
    
    residual = abs(unemp_change - (okun_alpha + okun_beta * output_growth))
    return residual <= okun_tau_sim

# 检查：工资通胀是否符合Phillips曲线预测
def check_phillips_consistency(unemployment, wage_inflation):
    if unemployment is None or wage_inflation is None:
        return True
    
    u_pct = unemployment * 100
    residual = abs(wage_inflation - (phillips_alpha + phillips_beta * u_pct))
    return residual <= phillips_tau_sim

# 筛选符合 Okun/Phillips 的好年份 
# 计算仿真数据的通胀分位数（用于替代 Real US 绝对范围）
all_price_infl = [m['price_inflation'] for m in year_metrics.values() if m['price_inflation'] is not None]
all_wage_infl = [m['wage_inflation'] for m in year_metrics.values() if m['wage_inflation'] is not None]

price_infl_p10 = np.percentile(all_price_infl, 10)
price_infl_p90 = np.percentile(all_price_infl, 90)
wage_infl_p10 = np.percentile(all_wage_infl, 10)
wage_infl_p90 = np.percentile(all_wage_infl, 90)

print(f"价格通胀范围: P10={price_infl_p10:.2f}%, P90={price_infl_p90:.2f}%")
print(f"工资通胀范围: P10={wage_infl_p10:.2f}%, P90={wage_infl_p90:.2f}%")
good_years = []

for year, metrics in year_metrics.items():
    output_growth = metrics['output_growth']
    price_inflation = metrics['price_inflation']
    wage_inflation = metrics['wage_inflation']  # 新增
    unemployment = metrics['unemployment']
    unemp_change = metrics['unemp_change']
    
    # 产出增长：不要太极端
    output_ok = output_growth is not None and (-10.0 <= output_growth <= 15.0)
    
    price_infl_ok = price_inflation is not None and (price_infl_p10 <= price_inflation <= price_infl_p90)
    
    # 工资通胀范围约束（用仿真数据分位数）
    wage_infl_ok = wage_inflation is not None and (wage_infl_p10 <= wage_inflation <= wage_infl_p90)
    
    # 失业率：不要超过 25%
    unemp_ok = unemployment is not None and unemployment <= 0.25
    
    # Okun/Phillips 方向一致性
    okun_ok = check_okun_consistency(output_growth, unemp_change)
    phillips_ok = check_phillips_consistency(unemployment, wage_inflation)  # 改用 wage_inflation
    
    is_good = output_ok and price_infl_ok and wage_infl_ok and unemp_ok and okun_ok and phillips_ok
    
    if is_good:
        good_years.append(year)
    
    status = "✓ GOOD" if is_good else "✗ BAD"
    fail_reasons = []
    if not output_ok: fail_reasons.append("产出范围")
    if not price_infl_ok: fail_reasons.append("价格通胀范围")
    if not wage_infl_ok: fail_reasons.append("工资通胀范围")
    if not unemp_ok: fail_reasons.append("失业率过高")
    if not okun_ok: fail_reasons.append("Okun方向")
    if not phillips_ok: fail_reasons.append("Phillips方向")
    
    print(f"Year {year}: {status} " + (f"(失败: {', '.join(fail_reasons)})" if fail_reasons else ""))

print(f"\n好年份: {good_years}")
print(f"好年份数量: {len(good_years)}/{len(year_metrics)}")

# 计算月度失业率分位数 

monthly_unemp_rates = get_monthly_unemployment_rates(states, max_t)
valid_rates = [r for r in monthly_unemp_rates if r is not None]

# 计算月度失业率的5%和95%分位数
unemp_p5 = np.percentile(valid_rates, 5)
unemp_p95 = np.percentile(valid_rates, 95)
print(f"月度失业率 P5: {unemp_p5*100:.2f}%")
print(f"月度失业率 P95: {unemp_p95*100:.2f}%")
print(f"使用区间: [{unemp_p5*100:.2f}%, {unemp_p95*100:.2f}%]")

# 从好年份中提取好月份

macro_good_months = []

for year in good_years:
    year_start = (year - 1) * 12
    year_end = year * 12
    
    for t in range(year_start, year_end):
        if t >= max_t:
            continue        
        monthly_unemployment = monthly_unemp_rates[t]
        if monthly_unemployment is None:
            continue
        if unemp_p5 <= monthly_unemployment <= unemp_p95:
            macro_good_months.append(t)

print(f"符合条件的月份数: {len(macro_good_months)}")

# 提取微观好决策 

good_decisions = []

# 预计算年度 MPC
agent_year_mpc = {}

for year in range(2, 21):
    year_start = (year - 1) * 12
    year_end = year * 12
    
    if year_end > max_t:
        break
    
    for agent_id_str in agent_ids:
        agent_id = int(agent_id_str) 
        
        yearly_dpi_change = 0
        yearly_c_change = 0
        
        for t in range(year_start, year_end):
            if t == 0 or agent_id_str not in states[t]:
                continue
            if periodic_tax[t].get(agent_id_str) is None or periodic_tax[t-1].get(agent_id_str) is None:
                continue

            
            curr_dpi = calculate_dpi(t, agent_id_str, states, periodic_tax)
            prev_dpi = calculate_dpi(t-1, agent_id_str, states, periodic_tax)
            
            curr_c = states[t][agent_id_str]['consumption']['Coin']
            prev_c = states[t-1][agent_id_str]['consumption']['Coin']
            
            yearly_dpi_change += (curr_dpi - prev_dpi)
            yearly_c_change += (curr_c - prev_c)
        
        if abs(yearly_dpi_change) > 500:
            agent_year_mpc[(agent_id_str, year)] = yearly_c_change / yearly_dpi_change
        else:
            agent_year_mpc[(agent_id_str, year)] = None
# ===== 检查 SimpleLabor 的真实取值范围 =====
print("\n=== 检查 SimpleLabor 取值范围 ===")
labor_vals = []
cons_vals = []
for t in range(1, max_t):
    for aid in agent_ids:
        a = actions[t].get(aid)
        if isinstance(a, dict):
            if 'SimpleLabor' in a:
                labor_vals.append(a['SimpleLabor'])
            if 'SimpleConsumption' in a:
                cons_vals.append(a['SimpleConsumption'])

print(f"SimpleLabor unique values: {sorted(set(labor_vals))[:20]}")
print(f"SimpleLabor min/max: {min(labor_vals)}, {max(labor_vals)}")
print(f"SimpleConsumption unique values: {sorted(set(cons_vals))[:20]}")
print(f"SimpleConsumption min/max: {min(cons_vals)}, {max(cons_vals)}")

# 从所有月份提取候选决策，用 reward 打分筛选
print("\n=== 使用 reward_function 打分筛选好决策 ===")

candidate_decisions = []

for t in range(1, max_t):  # 不再限制于 macro_good_months
    current_year = (t // 12) + 1
    
    for agent_id_str in agent_ids:
        agent_id = int(agent_id_str)
        
        if agent_id_str not in states[t] or agent_id_str not in states[t-1]:
            continue
        
        # ===== 获取 action =====
        if t >= len(actions) or agent_id_str not in actions[t]:
            continue
        
        action_data = actions[t][agent_id_str]
        
        # ===== 获取 action（只用来取消费；work 从 state['endogenous']['Labor'] 来）=====
        if isinstance(action_data, dict):
            if 'SimpleConsumption' not in action_data:
                continue
            consumption_idx = int(action_data['SimpleConsumption'])
        elif isinstance(action_data, (list, tuple)) and len(action_data) >= 1:
            consumption_idx = int(action_data[-1])
        else:
            continue

        # ===== work_decision：用 endogenous['Labor']/168（0 或 1）=====
        job = states[t][agent_id_str].get('endogenous', {}).get('job')
        if job == "Unemployment":
            work_decision = None   # 关键：失业 = 没有 work 决策
        else:
            work_decision = get_work_from_state(states[t][agent_id_str])

        consumption_idx = int(np.clip(consumption_idx, 0, 50))
        consumption_prop = consumption_idx * 0.02
        
        # ===== 基本可行性过滤（hard filter）=====
        curr_consumption = states[t][agent_id_str]['consumption']['Coin']
        curr_income = states[t][agent_id_str]['income']['Coin']
        curr_wealth = states[t][agent_id_str]['inventory']['Coin']
        
        curr_tax_data = periodic_tax[t].get(agent_id_str)
        if curr_tax_data is None:
            continue
        
        curr_lump = curr_tax_data.get('lump_sum', 0)
        curr_tax = curr_tax_data.get('tax_paid', 0)
        curr_dpi = curr_income + curr_lump - curr_tax
        
        # 物理约束：消费不能超过财富+收入太多
        if curr_consumption > curr_wealth + curr_income + 100:
            continue
        
        # ===== 构建 extra_info 并计算 reward =====
        extra_info = build_extra_info_for_reward(
            t, agent_id_str, states, actions, periodic_tax, prices, num_agents
        )
        
        # ===== 构建 solution_str（关键修改）=====
        if work_decision is None:
            # 失业：没有 work 这个 action
            solution_str = json.dumps({
                "consumption": consumption_prop
            })
        else:
            # 就业：有 work + consumption
            solution_str = json.dumps({
                "work": work_decision,
                "consumption": consumption_prop
            })
        
        # ===== 计算 reward =====
        reward_score = compute_score(
            data_source="econ_agent",
            solution_str=solution_str,
            ground_truth="",
            extra_info=extra_info
        )
        
        # ===== 收集候选 =====
        prev_consumption = states[t-1][agent_id_str]['consumption']['Coin']
        prev_income = states[t-1][agent_id_str]['income']['Coin']
        
        prev_tax_data = periodic_tax[t-1].get(agent_id_str, {})
        prev_lump = prev_tax_data.get('lump_sum', 0)
        prev_tax = prev_tax_data.get('tax_paid', 0)
        prev_dpi = prev_income + prev_lump - prev_tax
        
        job = states[t][agent_id_str].get('endogenous', {}).get('job')
        job_status = 0 if job == "Unemployment" else 1
        
        yearly_mpc = agent_year_mpc.get((agent_id_str, current_year), None)
        
        candidate_decisions.append({
            'timestep': t,
            'year': current_year,
            'agent_id': agent_id,
            'prev_consumption': prev_consumption,
            'curr_consumption': curr_consumption,
            'prev_income': prev_income,
            'curr_income': curr_income,
            'curr_wealth': curr_wealth,
            'prev_dpi': prev_dpi,
            'curr_dpi': curr_dpi,
            'work_decision': work_decision, 
            'job_status': job_status,        # 就业状态 0/1（用于统计）
            'consumption_prop': consumption_prop,
            'yearly_mpc': np.nan if yearly_mpc is None else yearly_mpc,
            'reward_score': reward_score,    # 用 reward 打分
            'r_okun_12m': extra_info.get('r_okun_12m'),
            'r_phil_12m': extra_info.get('r_phil_12m'),
            'extra_info_json': json.dumps(extra_info), 
        })

print(f"候选决策数: {len(candidate_decisions)}")

# ===== 按 reward_score 筛选 top 样本 =====
df_candidates = pd.DataFrame(candidate_decisions)

# 统计 reward_score 分布
print(f"\nreward_score 分布:")
print(f"  mean={df_candidates['reward_score'].mean():.4f}")
print(f"  std={df_candidates['reward_score'].std():.4f}")
print(f"  P25={df_candidates['reward_score'].quantile(0.25):.4f}")
print(f"  P50={df_candidates['reward_score'].quantile(0.50):.4f}")
print(f"  P75={df_candidates['reward_score'].quantile(0.75):.4f}")
print(f"  P90={df_candidates['reward_score'].quantile(0.90):.4f}")

# 分组计算 reward 阈值（关键修改）
employed = df_candidates[df_candidates['job_status'] == 1]
unemployed = df_candidates[df_candidates['job_status'] == 0]

thr_emp = employed['reward_score'].quantile(0.70)
thr_unemp = unemployed['reward_score'].quantile(0.40)  # 给失业更低门槛

df_good = pd.concat([
    employed[employed['reward_score'] >= thr_emp],
    unemployed[unemployed['reward_score'] >= thr_unemp]
], ignore_index=True)


# 重命名 reward_score 为 score（保持与原来的字段名兼容）
df_good['score'] = df_good['reward_score']
good_decisions = df_good.to_dict('records')

print(f"提取好决策数: {len(good_decisions)}")

# 平衡采样（基于 job_status，而不是 work_decision）
if len(good_decisions) == 0:
    print("\n警告: 没有找到符合条件的决策!")
    print("建议: 降低 reward_score 阈值")
else:
    df_good = pd.DataFrame(good_decisions)
    
    print("\n执行就业/失业平衡采样（基于 job_status）...")
    
    employed_decisions = df_good[df_good['job_status'] == 1].copy()
    unemployed_decisions = df_good[df_good['job_status'] == 0].copy()
    
    print(f"原始: 就业 {len(employed_decisions)}, 失业 {len(unemployed_decisions)}")
    
    target_unemployed_ratio = 0.15
    
    if len(unemployed_decisions) > 0 and len(employed_decisions) > 0:
        original_ratio = len(unemployed_decisions) / len(df_good)
        
        if original_ratio >= target_unemployed_ratio:
            print(f"失业比例已达标 ({original_ratio*100:.1f}%)")
        else:
            target_employed_count = int(len(unemployed_decisions) / target_unemployed_ratio * (1 - target_unemployed_ratio))
            
            if len(employed_decisions) > target_employed_count:
                # 按 reward_score 排序取 top
                employed_sampled = employed_decisions.nlargest(target_employed_count, 'reward_score')
            else:
                employed_sampled = employed_decisions
            
            df_good = pd.concat([employed_sampled, unemployed_decisions], ignore_index=True)
            print(f"平衡后: 就业 {len(employed_sampled)}, 失业 {len(unemployed_decisions)}")
    
    # 统计
    print("\n========== 数据质量检查 ==========")
    print(f"包含年份: {sorted(df_good['year'].unique())}")
    print(f"总决策数: {len(df_good)}")
    print(f"就业比例 (job_status): {(df_good['job_status']==1).sum()/len(df_good)*100:.1f}%")
    print(f"失业比例 (job_status): {(df_good['job_status']==0).sum()/len(df_good)*100:.1f}%")
    print(f"平均 reward_score: {df_good['reward_score'].mean():.4f}")
    print(f"work_decision 分布: mean={df_good['work_decision'].mean():.3f}, std={df_good['work_decision'].std():.3f}")# work_decision 可能有 None（失业样本）
    work_valid = df_good['work_decision'].dropna()
    if len(work_valid) > 0:
        print(f"work_decision 分布 (非空): n={len(work_valid)}, mean={work_valid.mean():.3f}, std={work_valid.std():.3f}")
    print(f"work_decision 为 None 的样本数: {df_good['work_decision'].isna().sum()}")    
    print(f"consumption_prop 分布: mean={df_good['consumption_prop'].mean():.3f}, std={df_good['consumption_prop'].std():.3f}")
    
    # 检查 r_okun/r_phil 分布
    r_okun_valid = df_good['r_okun_12m'].dropna()
    r_phil_valid = df_good['r_phil_12m'].dropna()
    if len(r_okun_valid) > 0:
        print(f"r_okun_12m: n={len(r_okun_valid)}, mean={r_okun_valid.mean():.3f}")
    if len(r_phil_valid) > 0:
        print(f"r_phil_12m: n={len(r_phil_valid)}, mean={r_phil_valid.mean():.3f}")
    
    # 保存
    output_path = '/workspace/QWEN2.5_42_7b_main/RL/good_decisions.csv'
    df_good.to_csv(output_path, index=False)
    print(f"\n保存到: {output_path}")
    