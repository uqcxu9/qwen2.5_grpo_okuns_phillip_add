# prepare_verl_data.py
import pickle as pkl
import pandas as pd
import os
import json
import numpy as np
import glob
from retrieval import GoodDecisionMemory, get_memory_context_for_prompt

# ==================== Macro variable calculation (aligned with reward_function.py) ====================
WINDOW = 24  # 2-year rolling window
GOOD_DECISION_MEMORY = None

def compute_trailing_12m_correlations(states, t, num_agents, dense_log):
    """
    计算 trailing 12 个月的 Okun/Phillips 相关系数
    修正：使用 lagged 窗口 [t-12, t-1] 避免泄漏
    """
    if t < 12:
        return None, None, None
    
    # 按配对方式收集（lagged 窗口：[t-12, t-1]）
    okun_pairs = []  # (du, gdp)
    phil_pairs = []  # (u, wage_infl) - 使用工资通胀
    u_values = []
    
    for tt in range(t - 12, t):  # [t-12, t-1]，不包含当期
        u = compute_unemployment_rate(states, tt, num_agents)
        du = compute_unemp_change(states, tt, num_agents)
        g = compute_gdp_growth(states, tt, num_agents)
        w_infl = compute_wage_inflation(states, tt, num_agents)  # Phillips 用工资通胀
        
        # Okun: 只有 du 和 g 都存在才配对
        if du is not None and g is not None:
            okun_pairs.append((du, g))
        
        # Phillips: 只有 u 和 w_infl 都存在才配对
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

def load_dialogs_from_files(data_dir):
    """
    从 dialog_*.pkl 文件加载对话历史，提取每个时间步每个 agent 的 prompt（最后的 user message）
    
    Returns:
        prompts: {t: {agent_id_str: prompt_text}}
    """
    prompts = {}
    
    # 查找所有 dialog 文件
    dialog_files = glob.glob(os.path.join(data_dir, "dialog_*.pkl"))
    
    for dialog_file in sorted(dialog_files):
        # 从文件名提取时间步，如 dialog_12.pkl -> t=12
        basename = os.path.basename(dialog_file)
        try:
            t = int(basename.replace("dialog_", "").replace(".pkl", ""))
        except:
            continue
        
        try:
            with open(dialog_file, 'rb') as f:
                dialog_data = pkl.load(f)
            
            prompts[t] = {}
            
            # dialog_data 可能是 list（每个元素是一个 agent 的对话 deque）
            # 或者是 dict（{agent_id: conversation_list}）
            if isinstance(dialog_data, list):
                # list of deques, index = agent_id
                for agent_idx, conversation in enumerate(dialog_data):
                    # conversation 是一个 deque，包含 {'role': 'user'/'assistant', 'content': ...}
                    if conversation:
                        # 提取最后一个 user message 作为 prompt
                        conv_list = list(conversation)
                        for msg in reversed(conv_list):
                            if isinstance(msg, dict) and msg.get('role') == 'user':
                                prompts[t][str(agent_idx)] = msg.get('content', '')
                                break
            elif isinstance(dialog_data, dict):
                for agent_id, conversation in dialog_data.items():
                    if conversation:
                        conv_list = list(conversation) if hasattr(conversation, '__iter__') else [conversation]
                        for msg in reversed(conv_list):
                            if isinstance(msg, dict) and msg.get('role') == 'user':
                                prompts[t][str(agent_id)] = msg.get('content', '')
                                break
        except Exception as e:
            print(f"[WARNING] Failed to load {dialog_file}: {e}")
            continue
    
    print(f"[Dialog] Loaded prompts from {len(prompts)} timesteps")
    return prompts


def get_good_decision_memory(csv_path=None):
    """Lazy load GoodDecisionMemory"""
    global GOOD_DECISION_MEMORY
    if GOOD_DECISION_MEMORY is None:
        if csv_path is None:
            csv_path = "/workspace/QWEN2.5_42_7b_main/RL/good_decisions.csv"
        if os.path.exists(csv_path):
            GOOD_DECISION_MEMORY = GoodDecisionMemory(csv_path)
            print(f"[Retrieval] Loaded {len(GOOD_DECISION_MEMORY.df)} good decision samples")
        else:
            print(f"[WARNING] good_decisions.csv not found: {csv_path}")
    return GOOD_DECISION_MEMORY

def compute_unemployment_rate(states, t, num_agents, agent_ids=None):
    """Calculate unemployment rate at time t"""
    if t < 0 or t >= len(states):
        return None
    unemployed = 0
    valid_count = 0
    
    # 优先用 agent_ids，否则回退 range
    ids_to_check = agent_ids if agent_ids else [str(i) for i in range(num_agents)]
    
    for agent_id_str in ids_to_check:
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


def compute_gdp_growth(states, t, num_agents):
    """Calculate GDP YoY growth rate (percentage)"""
    if t < 12 or t >= len(states):
        return None
    
    def get_total_income(states, t_idx):
        if t_idx < 0 or t_idx >= len(states):
            return None
        total = 0.0
        for i in range(num_agents):
            agent_state = states[t_idx].get(str(i), {})
            income = agent_state.get('income', {})
            if isinstance(income, dict):
                total += income.get('Coin', 0)
            else:
                total += float(income) if income else 0
        return total
    
    gdp_now = get_total_income(states, t)
    gdp_12m_ago = get_total_income(states, t - 12)
    
    if gdp_now is None or gdp_12m_ago is None or gdp_12m_ago < 1e-6:
        return None
    return (gdp_now - gdp_12m_ago) / gdp_12m_ago * 100


def compute_price_inflation(dense_log, t):
    """Calculate price inflation rate (annualized percentage)"""
    world = dense_log.get("world", [])
    prices = [None] * len(world)
    for i, w in enumerate(world):
        if isinstance(w, dict):
            prices[i] = w.get("Price")
    
    if not prices or t < 0 or t >= len(prices):
        return None
    
    if t < 12:
        if t >= 1:
            p_now = prices[t]
            p_prev = prices[t - 1]
            if p_now is None or p_prev is None:
                return None
            if p_prev > 1e-6:
                return (p_now - p_prev) / p_prev * 12 * 100
        return None
    
    p_now = prices[t]
    p_12m_ago = prices[t - 12]
    if p_now is None or p_12m_ago is None:
        return None
    if p_12m_ago < 1e-6:
        return None
    return (p_now - p_12m_ago) / p_12m_ago * 100


def compute_regime(states, t, num_agents, window=WINDOW):
    unemp_history, gdp_history = [], []
    for tt in range(max(1, t - window + 1), t + 1):
        u = compute_unemployment_rate(states, tt, num_agents)
        g = compute_gdp_growth(states, tt, num_agents)
        if u is not None:
            unemp_history.append(u)
        if g is not None:
            gdp_history.append(g)
    
    unemp_now = compute_unemployment_rate(states, t, num_agents)
    gdp_now = compute_gdp_growth(states, t, num_agents)
    
    if unemp_now is None or gdp_now is None:
        return None, None

    if len(unemp_history) < 3 or len(gdp_history) < 3:
        return None, None
    
    u_lo, u_hi = np.quantile(unemp_history, 0.20), np.quantile(unemp_history, 0.80)
    g_lo, g_hi = np.quantile(gdp_history, 0.20), np.quantile(gdp_history, 0.80)

    is_recession = (unemp_now >= u_hi) or (gdp_now <= g_lo)
    is_boom = (unemp_now <= u_lo) and (gdp_now >= g_hi)
    
    if is_recession and not is_boom:
        regime = "recession"
    elif is_boom and not is_recession:
        regime = "boom"
    else:
        regime = "normal"

    def clip01(x):
        return max(0.0, min(1.0, x))
    
    if regime == "recession":
        s_u = (unemp_now - u_hi) / (u_hi - u_lo + 1e-6) if u_hi > u_lo else 0.0
        s_g = (g_lo - gdp_now) / (g_hi - g_lo + 1e-6) if g_hi > g_lo else 0.0
    elif regime == "boom":
        s_u = (u_lo - unemp_now) / (u_hi - u_lo + 1e-6) if u_hi > u_lo else 0.0
        s_g = (gdp_now - g_hi) / (g_hi - g_lo + 1e-6) if g_hi > g_lo else 0.0
    else:
        s_u, s_g = 0.0, 0.0

    combined = 0.5 * clip01(s_g) + 0.5 * clip01(s_u)

    if regime == "normal":
        regime_strength = 0.15
    else:
        regime_strength = clip01(0.2 + 0.8 * combined)
    
    return regime, regime_strength


def compute_wage_inflation(states, t, num_agents):
    """Calculate wage inflation rate (YoY percentage)"""
    if t < 12 or t >= len(states):
        return None

    def avg_skill(t_idx):
        vals = []
        for i in range(num_agents):
            s = states[t_idx].get(str(i))
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


def compute_unemp_change(states, t, num_agents):
    """Unemployment rate change (YoY percentage points)"""
    if t < 12 or t >= len(states):
        return None
    u_now = compute_unemployment_rate(states, t, num_agents)
    u_12m = compute_unemployment_rate(states, t - 12, num_agents)
    if u_now is None or u_12m is None:
        return None
    return (u_now - u_12m) * 100.0


def _load_price_from_env(data_dir, dense_log):
    """Load price data from env_240.pkl and add to dense_log"""
    env_path = f"{data_dir}/env_240.pkl"
    if not os.path.exists(env_path):
        print(f"[ERROR] env_240.pkl not found: {env_path}")
        print(f"[ERROR] price_inflation will all be None")
        return dense_log
    
    class DummyUnpickler(pkl.Unpickler):
        def find_class(self, module, name):
            if 'ai_economist' in module:
                return type(name, (), {})
            return super().find_class(module, name)
    
    try:
        with open(env_path, "rb") as f:
            env = DummyUnpickler(f).load()
        
        prices = list(env.world.price)
        print(f"[Price] Loaded from env_240.pkl, {len(prices)} records")
        
        if "world" not in dense_log:
            dense_log["world"] = []
        
        while len(dense_log["world"]) < len(prices):
            dense_log["world"].append({})
        
        for i, p in enumerate(prices):
            if not isinstance(dense_log["world"][i], dict):
                dense_log["world"][i] = {}
            dense_log["world"][i]["Price"] = p
        
    except Exception as e:
        print(f"[ERROR] Failed to load env_240.pkl: {e}")
        print(f"[ERROR] price_inflation will all be None")
    
    return dense_log


SYSTEM_PROMPT = """You are an economic decision-making agent.

Based on your current financial situation, you should make economic decisions as follows:

- If you are employed, decide both:
  1. work: a value between 0 and 1 representing your labor supply
     (0 = no work, 1 = full-time work)
  2. consumption: a value between 0 and 1 representing the proportion of your disposable income to consume

- If you are unemployed, focus on the consumption decision.
  In this case, the work value will be ignored in evaluation.

You MUST respond with a valid JSON object in the exact format below:
{"work": <float between 0 and 1>, "consumption": <float between 0 and 1>}

Do not include any other text or explanation. Output only the JSON."""




def prepare_verl_dataset(data_dir, output_dir, num_agents=100):
    
    dense_log_path = f"{data_dir}/dense_log.pkl"
    with open(dense_log_path, 'rb') as f:
        dense_log = pkl.load(f)
    
    states = dense_log['states']
    actions = dense_log['actions']
    periodic_tax = dense_log['PeriodicTax']
    # 动态获取 agent_ids（新增）
    agent_ids = sorted([aid for aid in states[0].keys() if aid != "p" and isinstance(states[0][aid], dict)])
    actual_num_agents = len(agent_ids)
    print(f"[Info] Detected {actual_num_agents} agents: {agent_ids[:5]}...")
    # 从 dialog_*.pkl 文件加载 prompts
    prompts = load_dialogs_from_files(data_dir)
    
    # Check price data source
    if "world" in dense_log and len(dense_log["world"]) > 0:
        sample_world = dense_log["world"][0]
        if isinstance(sample_world, dict) and "Price" in sample_world:
            print(f"[Price] Loaded from dense_log['world'], {len(dense_log['world'])} records")
        else:
            print(f"[WARNING] dense_log['world'] exists but no Price field, trying env")
            dense_log = _load_price_from_env(data_dir, dense_log)
    else:
        print(f"[WARNING] dense_log missing 'world' field, trying env")
        dense_log = _load_price_from_env(data_dir, dense_log)
    
    # Sanity check: world length alignment
    if "world" in dense_log and len(dense_log["world"]) > 0:
        if len(dense_log["world"]) < len(states):
            print(f"[WARNING] world length ({len(dense_log['world'])}) < states length ({len(states)})")
            print(f"[WARNING] Some timesteps may have None price_inflation")
    
    samples = []
    
    errors = {
        'missing_state': 0,
        'missing_prompt': 0,
        'missing_action': 0,
        'invalid_action_format': 0,
        'invalid_skill': 0,
    }
    
    print("Pre-computing macro indicators...")
    macro_cache = {}
    
    u_hist, g_hist, pi_hist, w_hist, du_hist = [], [], [], [], []

    def rolling_ref_scale(hist, window=WINDOW):
        if len(hist) == 0:
            return None, None
        w = hist[-window:] if len(hist) >= window else hist
        ref = float(np.mean(w))
        scale = float(np.std(w))
        return ref, scale

    macro_cache[0] = {
        'unemployment_rate': None,
        'unemp_change': None,
        'gdp_growth': None,
        'price_inflation': None,
        'wage_inflation': None,
        'regime': None,
        'regime_strength': None,
        'unemp_ref': None,
        'gdp_ref': None,
        'infl_ref': None,
        'wage_infl_ref': None,
        'unemp_change_ref': None,
        'unemp_scale': None,
        'gdp_scale': None,
        'infl_scale': None,
        'wage_infl_scale': None,
        'unemp_change_scale': None,
    }

    for t in range(1, len(actions)):
        unemp = compute_unemployment_rate(states, t-1, num_agents)
        unemp_change = compute_unemp_change(states, t-1, num_agents)
        gdp_g = compute_gdp_growth(states, t-1, num_agents)
        infl = compute_price_inflation(dense_log, t-1)
        wage_infl = compute_wage_inflation(states, t-1, num_agents)

        if unemp is not None:
            u_hist.append(float(unemp))
        if gdp_g is not None:
            g_hist.append(float(gdp_g))
        if infl is not None:
            pi_hist.append(float(infl))
        if wage_infl is not None:
            w_hist.append(float(wage_infl))
        if unemp_change is not None:
            du_hist.append(float(unemp_change))

        unemp_ref, unemp_scale = rolling_ref_scale(u_hist, WINDOW)
        gdp_ref, gdp_scale = rolling_ref_scale(g_hist, WINDOW)
        infl_ref, infl_scale = rolling_ref_scale(pi_hist, WINDOW)
        wage_infl_ref, wage_infl_scale = rolling_ref_scale(w_hist, WINDOW)
        unemp_change_ref, unemp_change_scale = rolling_ref_scale(du_hist, WINDOW)

        regime, regime_strength = compute_regime(states, t-1, num_agents)

        macro_cache[t] = {
            'unemployment_rate': unemp,
            'unemp_change': unemp_change,
            'gdp_growth': gdp_g,
            'price_inflation': infl,
            'wage_inflation': wage_infl,
            'regime': regime,
            'regime_strength': regime_strength,
            'unemp_ref': unemp_ref,
            'gdp_ref': gdp_ref,
            'infl_ref': infl_ref,
            'wage_infl_ref': wage_infl_ref,
            'unemp_change_ref': unemp_change_ref,
            'unemp_scale': unemp_scale,
            'gdp_scale': gdp_scale,
            'infl_scale': infl_scale,
            'wage_infl_scale': wage_infl_scale,
            'unemp_change_scale': unemp_change_scale,
        }

    regime_counts = {
        r: sum(1 for m in macro_cache.values() if m.get('regime') == r)
        for r in ['normal', 'boom', 'recession']
    }
    none_regime = sum(1 for m in macro_cache.values() if m.get('regime') is None)
    print(f"Regime distribution: {regime_counts}, None={none_regime}")

    for t in range(1, len(actions)):
        for agent_id in range(num_agents):
            agent_id_str = str(agent_id)
            
            if agent_id_str not in states[t]:
                errors['missing_state'] += 1
                continue
            
            if t not in prompts or agent_id_str not in prompts[t]:
                errors['missing_prompt'] += 1
                continue
            
            raw_prompt = prompts[t][agent_id_str]
            if not raw_prompt or not isinstance(raw_prompt, str) or len(raw_prompt) < 100:
                errors['missing_prompt'] += 1
                continue
            
            state = states[t][agent_id_str]
            
            try:
                income = state['income']['Coin']
                wealth = state['inventory']['Coin']
                skill = state['skill']
            except (KeyError, TypeError):
                errors['missing_state'] += 1
                continue
            
            tax_info = periodic_tax[t].get(agent_id_str, {})
            tax_paid = tax_info.get('tax_paid', 0)
            lump_sum = tax_info.get('lump_sum', 0)
            
            dpi = income + lump_sum - tax_paid
            dpi_amt = max(dpi, 0.0)
            cash_on_hand = wealth + dpi_amt
            buffer_ratio = cash_on_hand / (dpi_amt + 1e-8) if dpi_amt > 1e-6 else 1.0

            action_data = actions[t].get(agent_id_str)
            if action_data is None:
                errors['missing_action'] += 1
                continue

            if isinstance(action_data, dict):
                labor_hours = state.get("endogenous", {}).get("Labor", None)
                if labor_hours is None:
                    work = None
                else:
                    try:
                        work = float(np.clip(float(labor_hours) / 168.0, 0.0, 1.0))
                    except:
                        work = None

                cons_idx = (
                    action_data.get('SimpleConsumption')
                    if 'SimpleConsumption' in action_data else
                    action_data.get('consumption')
                    if 'consumption' in action_data else
                    action_data.get('SimpleCons')
                    if 'SimpleCons' in action_data else
                    None
                )

                # 只要求 cons_idx 必须存在
                if cons_idx is None:
                    errors['invalid_action_format'] += 1
                    if errors['invalid_action_format'] <= 5:
                        print(
                            f"[invalid_action_format] t={t} agent={agent_id}: "
                            f"missing cons, keys={list(action_data.keys())}"
                        )
                    continue
                # work 可以为 None（失业时）
            elif isinstance(action_data, (list, tuple)) and len(action_data) >= 2:
                work = action_data[0]
                cons_idx = action_data[1]
            else:
                errors['invalid_action_format'] += 1
                if errors['invalid_action_format'] <= 5:
                    print(f"[invalid_action_format] t={t} agent={agent_id}: invalid format type={type(action_data).__name__} value={repr(action_data)[:100]}")
                continue

            consumption_prop = float(np.clip(cons_idx * 0.02, 0.0, 1.0))

            job_label = state.get("endogenous", {}).get("job", "Unknown")  # 字符串
            gt_employed = 1 if job_label != "Unemployment" else 0
            job_status = gt_employed  # 数值 0/1，与 good_decisions.csv 一致
            
            # Build few-shot context
            memory = get_good_decision_memory()
            few_shot_context = ""
            
            if memory is not None:
                # Get previous consumption
                prev_consumption = 0.0
                if t > 0 and agent_id_str in states[t-1]:
                    prev_cons_data = states[t-1][agent_id_str].get('consumption', {})
                    if isinstance(prev_cons_data, dict):
                        prev_consumption = prev_cons_data.get('Coin', 0.0)
                    else:
                        prev_consumption = float(prev_cons_data) if prev_cons_data else 0.0
                
                # Get previous income
                prev_income = 0.0
                if t > 0 and agent_id_str in states[t-1]:
                    prev_income_data = states[t-1][agent_id_str].get('income', {})
                    if isinstance(prev_income_data, dict):
                        prev_income = prev_income_data.get('Coin', 0.0)
                    else:
                        prev_income = float(prev_income_data) if prev_income_data else 0.0
                
                # Get previous DPI
                prev_dpi = 0.0
                if t > 0:
                    prev_tax_info = periodic_tax[t-1].get(agent_id_str, {})
                    prev_tax_paid = prev_tax_info.get('tax_paid', 0.0)
                    prev_lump_sum = prev_tax_info.get('lump_sum', 0.0)
                    prev_dpi = prev_income + prev_lump_sum - prev_tax_paid
                
                # Build current state dict
                current_state = {
                    'income': float(income),
                    'wealth': float(wealth),
                    'dpi': float(dpi),
                    'prev_consumption': float(prev_consumption),
                    'prev_income': float(prev_income),
                    'prev_dpi': float(prev_dpi),
                    'job': job_status,
                }
                
                # Retrieve and format
                few_shot_context = get_memory_context_for_prompt(current_state, memory, k=3)
            
            # Prepend few-shot context to raw_prompt
            if few_shot_context:
                enhanced_prompt = few_shot_context + "\n\n" + raw_prompt
            else:
                enhanced_prompt = raw_prompt
            
            # Build verl expected chat format prompt
            chat_prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": enhanced_prompt}
            ]
            
            buffer_ratio = max(0.0, min(10.0, buffer_ratio))
            macro = macro_cache[t]
            r_okun_12m, r_phil_12m, u_mean_12m = compute_trailing_12m_correlations(
                states, t, num_agents, dense_log
            )     
            gt_labor = None if work is None else float(work)       
            extra_info_dict = {
                "timestep": t,
                "agent_id": agent_id,
                "unemp_change": macro.get("unemp_change", None),
                "income": float(income),
                "wealth": float(wealth),
                "dpi": float(dpi),
                "buffer_ratio": float(buffer_ratio),
                "tax_paid": float(tax_paid),
                "lump_sum": float(lump_sum),
                "skill": float(skill),
                "gt_labor_supply": gt_labor,
                "gt_labor_supply_bin": None if gt_labor is None else int(gt_labor >= 0.5),
                "gt_employed": gt_employed,
                "job_status": job_status,
                "job_label": job_label,
                "gt_consumption": float(consumption_prop),
                "unemployment_rate": macro['unemployment_rate'],
                "gdp_growth": macro['gdp_growth'],
                "price_inflation": macro['price_inflation'],
                "wage_inflation": macro.get('wage_inflation', None),
                "regime": macro['regime'],
                "regime_strength": macro['regime_strength'],
                "unemp_ref": macro.get('unemp_ref', None),
                "gdp_ref": macro.get('gdp_ref', None),
                "infl_ref": macro.get('infl_ref', None),
                "wage_infl_ref": macro.get('wage_infl_ref', None),
                "unemp_change_ref": macro.get('unemp_change_ref', None),
                "unemp_scale": macro.get('unemp_scale', None),
                "gdp_scale": macro.get('gdp_scale', None),
                "infl_scale": macro.get('infl_scale', None),
                "wage_infl_scale": macro.get('wage_infl_scale', None),
                "unemp_change_scale": macro.get('unemp_change_scale', None),
                "r_okun_12m": r_okun_12m,
                "r_phil_12m": r_phil_12m,
                "u_mean_12m": u_mean_12m,
            }
            
            samples.append({
                "prompt": chat_prompt,
                "data_source": "econ_agent",
                "extra_info": extra_info_dict,
                "reward_model": {"ground_truth": ""},
                "ability": "economic_decision",
            })
    
    print(f"\n=== Data Processing Statistics ===")
    print(f"Successful samples: {len(samples)}")
    print(f"Error statistics:")
    for err_type, count in errors.items():
        if count > 0:
            print(f"  - {err_type}: {count}")
    total_errors = sum(errors.values())
    print(f"  Total errors: {total_errors}")
    
    if len(samples) == 0:
        raise ValueError("No valid samples! Please check data source.")
    
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(samples)
    
    all_agents = list(range(num_agents))
    np.random.seed(42)
    val_agents = set(np.random.choice(all_agents, size=5, replace=False))
    train_agents = set(all_agents) - val_agents
    
    print(f"\nValidation agents: {sorted(val_agents)}")
    print(f"Training agents count: {len(train_agents)}")
    
    train_df = df[df['extra_info'].apply(lambda x: x['agent_id'] in train_agents)]
    val_df_full = df[df['extra_info'].apply(lambda x: x['agent_id'] in val_agents)]
    
    val_size = min(200, len(val_df_full))
    val_df = val_df_full.sample(n=val_size, random_state=42)
    
    # 保存为 JSONL 格式（VERL 常用格式，无需额外依赖）
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_json(f"{output_dir}/train.jsonl", orient='records', lines=True, force_ascii=False)
    val_df.to_json(f"{output_dir}/val.jsonl", orient='records', lines=True, force_ascii=False)
    
    print(f"\nSaved {len(train_df)} training, {len(val_df)} validation samples")
    print(f"   Validation set sampled {val_size} from {len(val_df_full)} samples")
    print(f"   Output directory: {output_dir}")
    
    print(f"\n=== Data Format Validation ===")
    print(f"prompt type: {type(samples[0]['prompt'])}")
    print(f"prompt length: {len(samples[0]['prompt'])} messages")
    print(f"prompt[0] role: {samples[0]['prompt'][0]['role']}")
    print(f"prompt[1] role: {samples[0]['prompt'][1]['role']}")
    print(f"user content first 100 chars: {samples[0]['prompt'][1]['content'][:100]}...")
    
    # ===== Buffer Ratio 分布统计 =====
    print(f"\n=== Buffer Ratio Distribution ===")
    br_values = [s['extra_info']['buffer_ratio'] for s in samples if 'buffer_ratio' in s['extra_info']]
    if br_values:
        print(f"Buffer Ratio: mean={np.mean(br_values):.2f}, std={np.std(br_values):.2f}")
        print(f"  Q10={np.percentile(br_values, 10):.2f}, Q25={np.percentile(br_values, 25):.2f}")
        print(f"  Q50={np.percentile(br_values, 50):.2f}, Q75={np.percentile(br_values, 75):.2f}")
        print(f"  Q90={np.percentile(br_values, 90):.2f}")
        
        # 当前 BR_LO/BR_HI 覆盖多少数据
        BR_LO, BR_HI = 1.7505, 2.5063
        in_range = sum(1 for br in br_values if BR_LO <= br <= BR_HI)
        print(f"  In [BR_LO={BR_LO}, BR_HI={BR_HI}]: {in_range}/{len(br_values)} = {in_range/len(br_values)*100:.1f}%")
    
    # ===== 年度 Bonus 触发频率统计 =====
    print(f"\n=== Yearly Bonus Statistics ===")
    bonus_samples = [s for s in samples if s['extra_info'].get('timestep', 0) >= 12 and s['extra_info'].get('timestep', 0) % 12 == 0]
    print(f"Yearly bonus samples: {len(bonus_samples)}/{len(samples)} = {len(bonus_samples)/len(samples)*100:.1f}%")
    
    if bonus_samples:
        # 目标相关系数和容忍度（与 reward.py 保持一致）
        OKUN_TGT = -0.894
        PHIL_TGT = -0.676
        TOL_OKUN = 0.25
        TOL_PHIL = 0.25
        
        # r_okun_12m 分布
        r_okun_values = [s['extra_info'].get('r_okun_12m') for s in bonus_samples if s['extra_info'].get('r_okun_12m') is not None]
        if r_okun_values:
            okun_pass = sum(1 for r in r_okun_values if abs(r - OKUN_TGT) <= TOL_OKUN)
            print(f"r_okun_12m: mean={np.mean(r_okun_values):.3f}, "
                  f"pass rate (within {TOL_OKUN} of {OKUN_TGT}): "
                  f"{okun_pass}/{len(r_okun_values)} = {okun_pass/len(r_okun_values)*100:.1f}%")
        else:
            print("r_okun_12m: No data available")
        
        # r_phil_12m 分布
        r_phil_values = [s['extra_info'].get('r_phil_12m') for s in bonus_samples if s['extra_info'].get('r_phil_12m') is not None]
        if r_phil_values:
            phil_pass = sum(1 for r in r_phil_values if abs(r - PHIL_TGT) <= TOL_PHIL)
            print(f"r_phil_12m: mean={np.mean(r_phil_values):.3f}, "
                  f"pass rate (within {TOL_PHIL} of {PHIL_TGT}): "
                  f"{phil_pass}/{len(r_phil_values)} = {phil_pass/len(r_phil_values)*100:.1f}%")
        else:
            print("r_phil_12m: No data available")
        
        # u_mean_12m 分布
        u_mean_values = [s['extra_info'].get('u_mean_12m') for s in bonus_samples if s['extra_info'].get('u_mean_12m') is not None]
        if u_mean_values:
            u_pass = sum(1 for u in u_mean_values if 0.04 <= u <= 0.15)
            print(f"u_mean_12m: mean={np.mean(u_mean_values):.3f}, pass rate (0.04-0.15): {u_pass}/{len(u_mean_values)} = {u_pass/len(u_mean_values)*100:.1f}%")
        else:
            print("u_mean_12m: No data available")
    
    return train_df, val_df


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    data_dir = os.path.join(script_dir, "/workspace/QWEN2.5_42_7b_main/data/gpt-3-noperception-reflection-1-100agents-240months")
    output_dir = os.path.join(script_dir, "verl_dataset")
    
    prepare_verl_dataset(data_dir, output_dir)