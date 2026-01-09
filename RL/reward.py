import json
import re
import math
import random

_DEBUG_COUNT = 0
_DEFAULT_VALUE_COUNT = 0
_SEED_SET = False


def range_reward(x: float, low: float, high: float) -> float:
    width = max(high - low, 1e-6)
    
    if x < low:
        return -min((low - x) / width, 1.0)
    elif x > high:
        return -min((x - high) / width, 1.0)
    else:
        mid = 0.5 * (low + high)
        half = max(0.5 * width, 1e-6)
        val = 1.0 - 2.0 * (abs(x - mid) / half)
        return max(-1.0, min(1.0, val))


def parse_action(response: str):
    try:
        text = response.replace('```json', '').replace('```', '').strip()
        decoder = json.JSONDecoder()
        idx = text.find('{')
        if idx == -1:
            return None
        obj, end = decoder.raw_decode(text, idx)
        return obj if isinstance(obj, dict) else None
    except:
        return None


def _to_float_or_none(x):
    if x is None:
        return None
    try:
        x = float(x)
    except:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x
def target_corr_reward(r, r_target, tol=0.25):
    # tol：允许偏差，0.25 意味着差 0.25 就降到 0
    if r is None:
        return 0.0
    err = abs(r - r_target)
    score = 1.0 - err / tol
    return max(-1.0, min(1.0, score))


def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs) -> float:
    global _DEBUG_COUNT, _DEFAULT_VALUE_COUNT, _SEED_SET
    if not _SEED_SET:
        random.seed(42)
        _SEED_SET = True
    _DEBUG_COUNT += 1

    if _DEBUG_COUNT <= 5:
        try:
            with open('/workspace/reward_debug.log', 'a') as f:
                f.write(f"\n=== DEBUG {_DEBUG_COUNT} ===\n")
                f.write(f"solution_str: {repr(solution_str)[:500]}\n")
                f.write(f"extra_info: {repr(extra_info)[:300]}\n")
        except:
            pass

    if data_source != "econ_agent":
        return 0.0

    reward = 0.0

    action = parse_action(solution_str)
    if action is None:
        return -1.0

    # === 强格式检查（GRPO 必须严格） ===
    if not isinstance(action, dict):
        return -1.0

    if "work" not in action or "consumption" not in action:
        return -1.0

    # 严格 float 转换
    try:
        work = float(action["work"])
        consumption = float(action["consumption"])
    except:
        return -1.0

    # 越界直接 -1.0（不是软惩罚）
    if not (0 <= work <= 1):
        return -1.0
    if not (0 <= consumption <= 1):
        return -1.0

    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except:
            extra_info = {}
    elif extra_info is None:
        extra_info = {}

    income = _to_float_or_none(extra_info.get('income', None))
    lump_sum = _to_float_or_none(extra_info.get('lump_sum', None))
    tax_paid = _to_float_or_none(extra_info.get('tax_paid', None))
    wealth = _to_float_or_none(extra_info.get('wealth', None))
    
    missing_micro = []
    if income is None:
        missing_micro.append('income')
        income = 0.0
    if lump_sum is None:
        missing_micro.append('lump_sum')
        lump_sum = 0.0
    if tax_paid is None:
        missing_micro.append('tax_paid')
        tax_paid = 0.0
    if wealth is None:
        missing_micro.append('wealth')
        wealth = 0.0

    dpi = _to_float_or_none(extra_info.get('dpi', None))
    if dpi is None:
        dpi = income + lump_sum - tax_paid
    dpi = float(dpi)

    buffer_ratio = _to_float_or_none(extra_info.get('buffer_ratio', None))
    if buffer_ratio is None:
        if dpi > 1e-6:
            cash_on_hand = wealth + dpi
            buffer_ratio = cash_on_hand / (dpi + 1e-8)
        else:
            buffer_ratio = 1.0
    buffer_ratio = max(0.0, min(10.0, float(buffer_ratio)))

    unemp = _to_float_or_none(extra_info.get('unemployment_rate', None))
    gdp_g = _to_float_or_none(extra_info.get('gdp_growth', None))
    unemp_change = _to_float_or_none(extra_info.get('unemp_change', None))
    infl = _to_float_or_none(extra_info.get('price_inflation', None))
    wage_infl = _to_float_or_none(extra_info.get('wage_inflation', None))

    regime = extra_info.get("regime", None)
    if regime not in ("normal", "boom", "recession"):
        regime = None

    regime_strength = _to_float_or_none(extra_info.get("regime_strength", None))
    if regime_strength is not None:
        regime_strength = max(0.0, min(1.0, float(regime_strength)))

# ===== 微观目标对齐（让 reward 依赖 action）=====
    
    gt_employed = extra_info.get("gt_employed", None)
    
    # Consumption: 根据 buffer_ratio 构造目标
    br = buffer_ratio
    cons_target = 0.25 + 0.50 * (1.0 - 1.0 / (1.0 + br))
    cons_target = max(0.10, min(0.85, cons_target))
    cons_r = 1.0 - 2.0 * abs(consumption - cons_target)
    cons_r = max(-1.0, min(1.0, cons_r))
    
    # Work: 让 work_target 随 buffer_ratio 变化
    if gt_employed == 1:
        work_target = 0.95 - 0.08 * min(br, 6.0)  # br=0→0.95, br=6→0.47
        work_target = max(0.4, min(0.95, work_target))
        
        # 使用 range_reward 而不是简单 L1，让梯度更平滑
        # 允许 work 在 [target-0.15, target+0.15] 范围内
        work_r = range_reward(work, work_target - 0.15, work_target + 0.15)
        
        micro_r = 0.3 * work_r + 0.7 * cons_r
    else:
        # 失业：不计 work_r
        micro_r = cons_r

    # === Macro guard ===
    guard_parts = []
    guard_w = []

    if unemp is not None:
        guard_parts.append(range_reward(unemp, 0.02, 0.20))
        guard_w.append(0.40)
    if gdp_g is not None:
        guard_parts.append(range_reward(gdp_g, -5.0, 10.0))
        guard_w.append(0.30)
    if wage_infl is not None:
        guard_parts.append(range_reward(wage_infl, -3.4, 4.5))
        guard_w.append(0.20)
    if infl is not None:
        guard_parts.append(range_reward(infl, -5.5, 8.8))
        guard_w.append(0.10)
    if guard_w:
        wsum = sum(guard_w)
        guard = sum(p * w for p, w in zip(guard_parts, guard_w)) / wsum
    else:
        guard = 0.0
    guard = max(-1.0, min(1.0, guard))

    # === 动作正则（只惩罚消费极端值）===
    reg_penalty = 0.0
    # work 是离散动作 (0 或 1)，不在这里惩罚
    if consumption < 0.05 or consumption > 0.95:
        reg_penalty -= 0.5
    reg_penalty = max(-1.0, min(0.0, reg_penalty))

    # === 年度宏观拟合 ===
    r_okun_12m = _to_float_or_none(extra_info.get("r_okun_12m", None))
    r_phil_12m = _to_float_or_none(extra_info.get("r_phil_12m", None))
    u_mean_12m = _to_float_or_none(extra_info.get("u_mean_12m", None))

    # 目标相关系数（来自真实经济数据）
    R_OKUN_TARGET = -0.894
    R_PHIL_TARGET = -0.676
    
    okun_score = target_corr_reward(r_okun_12m, R_OKUN_TARGET, tol=0.25)
    phil_score = target_corr_reward(r_phil_12m, R_PHIL_TARGET, tol=0.25)

    if u_mean_12m is not None:
        u_score = range_reward(u_mean_12m, 0.04, 0.15)
    else:
        u_score = 0.0

    annual_fit = 0.4 * okun_score + 0.4 * phil_score + 0.2 * u_score
    annual_fit = max(-1.0, min(1.0, annual_fit))

# === Final combine（提升 micro 权重，让 action 影响 reward）===
    has_annual = (r_okun_12m is not None) or (r_phil_12m is not None)

    if has_annual:
        W_ANNUAL = 0.45
        W_GUARD = 0.30
        W_MICRO = 0.20
        W_REG = 0.05
    else:
        W_ANNUAL = 0.00
        W_GUARD = 0.65
        W_MICRO = 0.30
        W_REG = 0.05

    reward = 0.0
    reward += W_ANNUAL * annual_fit
    reward += W_GUARD * guard
    reward += W_MICRO * micro_r
    reward += W_REG * reg_penalty

    reward = max(-1.0, min(1.0, reward))
    return reward