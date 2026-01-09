"""
训练结果分析脚本 v2
- 统计解析成功率
- 保存原始 LoRA 输出
- 使用 temperature=0.0 确定性输出
"""
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
sys.path.append('/workspace/QWEN2.5_42_7b_main/RL')
from reward import compute_score

# ===== 配置 =====
BASE_MODEL_PATH = "/workspace/models/Qwen2.5-7B-Instruct"
LORA_PATH = "/workspace/QWEN2.5_42_7b_main/checkpoints/global_step_100/actor/lora_adapter"
VAL_DATA_PATH = "/workspace/QWEN2.5_42_7b_main/RL/verl_dataset/val.parquet"
NUM_SAMPLES = 50  # 先跑 50 条快速检查

def extract_action_v2(response_text):
    """
    从模型输出中提取 action
    返回 (action, ok, reason, raw_json)
    """
    original = response_text
    response_text = response_text.strip()
    
    # 尝试各种方式解析
    reasons = []
    
    # 方式1: 直接解析整个响应
    try:
        action = json.loads(response_text)
        if 'work' in action and 'consumption' in action:
            return {
                "work": float(action['work']),
                "consumption": float(action['consumption'])
            }, True, "direct_json", response_text
    except:
        reasons.append("not_direct_json")
    
    # 方式2: 查找 ```json ``` 块
    if "```json" in response_text:
        try:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
            action = json.loads(json_str)
            if 'work' in action and 'consumption' in action:
                return {
                    "work": float(action['work']),
                    "consumption": float(action['consumption'])
                }, True, "json_block", json_str
        except:
            reasons.append("json_block_parse_failed")
    
    # 方式3: 查找 ``` ``` 块
    if "```" in response_text and "```json" not in response_text:
        try:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
            action = json.loads(json_str)
            if 'work' in action and 'consumption' in action:
                return {
                    "work": float(action['work']),
                    "consumption": float(action['consumption'])
                }, True, "code_block", json_str
        except:
            reasons.append("code_block_parse_failed")
    
    # 方式4: 查找 { } 包围的部分
    start = response_text.find('{')
    end = response_text.rfind('}') + 1
    if start != -1 and end > start:
        json_str = response_text[start:end]
        try:
            action = json.loads(json_str)
            work = action.get('work', action.get('work_decision', action.get('Work', None)))
            consumption = action.get('consumption', action.get('consumption_prop', action.get('Consumption', None)))
            if work is not None and consumption is not None:
                return {
                    "work": float(work),
                    "consumption": float(consumption)
                }, True, "extracted_json", json_str
        except json.JSONDecodeError as e:
            reasons.append(f"json_decode_error: {str(e)[:50]}")
    else:
        reasons.append("no_braces_found")
    
    # 解析失败，返回默认值
    reason = "; ".join(reasons) if reasons else "unknown"
    return {"work": 0.5, "consumption": 0.5}, False, reason, ""

def main():
    print("=" * 70)
    print("训练结果分析 v2 (详细诊断)")
    print("=" * 70)
    
    # ===== 1. 加载模型 =====
    print("\n[1/4] 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()
    print("  模型加载完成!")
    
    # ===== 2. 加载验证集 =====
    print("\n[2/4] 加载验证集...")
    val_df = pd.read_parquet(VAL_DATA_PATH)
    if len(val_df) > NUM_SAMPLES:
        val_df = val_df.sample(n=NUM_SAMPLES, random_state=42)
    print(f"  验证集: {len(val_df)} 条")
    
    # ===== 3. 生成并评估 =====
    print("\n[3/4] 生成 action (temperature=0.0, do_sample=False)...")
    
    results = []
    parse_stats = {"ok": 0, "fail": 0}
    fail_reasons = {}
    raw_outputs = []
    
    def generate_action(model_obj, text, use_adapter=True):
        """使用 temperature=0.0, do_sample=False 确定性生成"""
        if not use_adapter:
            with model_obj.disable_adapter():
                inputs = tokenizer(text, return_tensors="pt").to(model_obj.device)
                with torch.no_grad():
                    outputs = model_obj.generate(
                        **inputs,
                        max_new_tokens=128,
                        temperature=0.0,  # 确定性
                        do_sample=False,  # 不采样
                        pad_token_id=tokenizer.pad_token_id
                    )
                response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        else:
            inputs = tokenizer(text, return_tensors="pt").to(model_obj.device)
            with torch.no_grad():
                outputs = model_obj.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.0,  # 确定性
                    do_sample=False,  # 不采样
                    pad_token_id=tokenizer.pad_token_id
                )
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        action, ok, reason, raw_json = extract_action_v2(response)
        return action, response, ok, reason, raw_json
    
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df)):
        prompt_messages = row["prompt"]
        extra_info = row.get("extra_info", {})
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except:
                extra_info = {}
        elif extra_info is None:
            extra_info = {}
        
        text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        
        # LoRA model (确定性)
        lora_action, lora_response, lora_ok, lora_reason, lora_raw_json = generate_action(model, text, use_adapter=True)
        lora_reward = compute_score("econ_agent", json.dumps(lora_action), "", extra_info)
        
        # 统计解析
        if lora_ok:
            parse_stats["ok"] += 1
        else:
            parse_stats["fail"] += 1
            fail_reasons[lora_reason] = fail_reasons.get(lora_reason, 0) + 1
        
        # 保存原始输出
        raw_outputs.append({
            'idx': idx,
            'timestep': extra_info.get('timestep', 'N/A'),
            'agent_id': extra_info.get('agent_id', 'N/A'),
            'lora_response': lora_response[:500],  # 前 500 字符
            'lora_ok': lora_ok,
            'lora_reason': lora_reason,
            'lora_raw_json': lora_raw_json[:200] if lora_raw_json else "",
            'lora_work': lora_action['work'],
            'lora_consumption': lora_action['consumption'],
            'lora_reward': lora_reward,
        })
        
        results.append({
            'timestep': extra_info.get('timestep', 'N/A'),
            'agent_id': extra_info.get('agent_id', 'N/A'),
            'lora_work': lora_action['work'],
            'lora_consumption': lora_action['consumption'],
            'lora_reward': lora_reward,
            'lora_ok': lora_ok,
            'lora_reason': lora_reason,
        })
    
    # ===== 4. 输出分析结果 =====
    print("\n" + "=" * 70)
    print("[4/4] 分析结果")
    print("=" * 70)
    
    # 1. 解析成功率
    print("\n1. 解析成功率")
    print("-" * 40)
    total = parse_stats["ok"] + parse_stats["fail"]
    print(f"  成功: {parse_stats['ok']}/{total} ({100*parse_stats['ok']/total:.1f}%)")
    print(f"  失败: {parse_stats['fail']}/{total} ({100*parse_stats['fail']/total:.1f}%)")
    
    if fail_reasons:
        print("\n  失败原因分布:")
        for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")
    
    # 2. Work/Consumption 分布（仅成功解析的）
    results_df = pd.DataFrame(results)
    ok_df = results_df[results_df['lora_ok'] == True]
    
    print("\n2. Work/Consumption 分布 (成功解析的)")
    print("-" * 40)
    if len(ok_df) > 0:
        print(f"  Work: mean={ok_df['lora_work'].mean():.3f}, std={ok_df['lora_work'].std():.3f}")
        print(f"  Consumption: mean={ok_df['lora_consumption'].mean():.3f}, std={ok_df['lora_consumption'].std():.3f}")
        print(f"\n  Work 分布:")
        print(f"    [0, 0.2]: {(ok_df['lora_work'] < 0.2).sum()}")
        print(f"    [0.2, 0.4]: {((ok_df['lora_work'] >= 0.2) & (ok_df['lora_work'] < 0.4)).sum()}")
        print(f"    [0.4, 0.6]: {((ok_df['lora_work'] >= 0.4) & (ok_df['lora_work'] < 0.6)).sum()}")
        print(f"    [0.6, 0.8]: {((ok_df['lora_work'] >= 0.6) & (ok_df['lora_work'] < 0.8)).sum()}")
        print(f"    [0.8, 1.0]: {(ok_df['lora_work'] >= 0.8).sum()}")
    else:
        print("  没有成功解析的样本!")
    
    # 3. 原始 LoRA 输出（前 20 条）
    print("\n3. 原始 LoRA 输出 (前 20 条)")
    print("=" * 70)
    
    for i, out in enumerate(raw_outputs[:20]):
        print(f"\n--- 样本 {i+1} (t={out['timestep']}, agent={out['agent_id']}) ---")
        print(f"解析: {'✅ OK' if out['lora_ok'] else '❌ FAIL'} ({out['lora_reason']})")
        print(f"提取结果: work={out['lora_work']}, consumption={out['lora_consumption']}")
        print(f"原始输出 (前 300 字符):")
        print(f"  \"{out['lora_response'][:300]}\"")
        if out['lora_raw_json']:
            print(f"解析到的 JSON: {out['lora_raw_json']}")
    
    # 保存原始输出
    raw_df = pd.DataFrame(raw_outputs)
    raw_df.to_csv('/workspace/QWEN2.5_42_7b_main/lora_raw_outputs.csv', index=False)
    print(f"\n原始输出已保存到: /workspace/QWEN2.5_42_7b_main/lora_raw_outputs.csv")
    
    # 结论
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    
    if parse_stats["fail"] / total > 0.3:
        print(f"⚠️ 解析失败率过高 ({100*parse_stats['fail']/total:.1f}%)，可能是格式问题而非 collapse")
    
    if len(ok_df) > 0 and ok_df['lora_work'].std() < 0.01:
        print(f"❌ Work 几乎无方差 (std={ok_df['lora_work'].std():.4f})，确认 mode collapse")
    elif len(ok_df) > 0:
        print(f"✅ Work 有多样性 (std={ok_df['lora_work'].std():.4f})")
    
    return raw_df

if __name__ == "__main__":
    main()
