"""
训练结果分析脚本
分析：KL, reward分布, buffer ratio关系, work分布, 宏观关系检查
"""
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
# import matplotlib.pyplot as plt  # 不需要
import sys
sys.path.append('/workspace/QWEN2.5_42_7b_main/RL')
from reward import compute_score

# ===== 配置 =====
BASE_MODEL_PATH = "/workspace/models/Qwen2.5-7B-Instruct"
LORA_PATH = "/workspace/QWEN2.5_42_7b_main/checkpoints/global_step_1000/actor/lora_adapter"
VAL_DATA_PATH = "/workspace/QWEN2.5_42_7b_main/RL/verl_dataset/val.parquet"
NUM_SAMPLES = 195  # 全部验证集

def extract_action(response_text):
    """从模型输出中提取 action"""
    try:
        response_text = response_text.strip()
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start != -1 and end > start:
            json_str = response_text[start:end]
            action = json.loads(json_str)
            work = action.get('work', action.get('work_decision', action.get('Work', 0.5)))
            consumption = action.get('consumption', action.get('consumption_prop', action.get('Consumption', 0.5)))
            return {"work": float(work), "consumption": float(consumption)}
    except:
        pass
    return {"work": 0.5, "consumption": 0.5}

def main():
    print("=" * 70)
    print("训练结果分析 (Step 1000)")
    print("=" * 70)
    
    # ===== 1. 加载模型 =====
    print("\n[1/6] 加载模型...")
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
    print("\n[2/6] 加载验证集...")
    val_df = pd.read_parquet(VAL_DATA_PATH)
    if len(val_df) > NUM_SAMPLES:
        val_df = val_df.sample(n=NUM_SAMPLES, random_state=42)
    print(f"  验证集: {len(val_df)} 条")
    
    # ===== 3. 生成并评估 =====
    print("\n[3/6] 生成 action 并评估...")
    
    results = []
    base_rewards = []
    lora_rewards = []
    random_rewards = []
    
    def generate_action(model_obj, text, use_adapter=True):
        if not use_adapter:
            with model_obj.disable_adapter():
                inputs = tokenizer(text, return_tensors="pt").to(model_obj.device)
                with torch.no_grad():
                    outputs = model_obj.generate(**inputs, max_new_tokens=128, temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=tokenizer.pad_token_id)
                response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        else:
            inputs = tokenizer(text, return_tensors="pt").to(model_obj.device)
            with torch.no_grad():
                outputs = model_obj.generate(**inputs, max_new_tokens=128, temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=tokenizer.pad_token_id)
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return extract_action(response), response
    
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
        
        # Base model
        base_action, _ = generate_action(model, text, use_adapter=False)
        base_reward = compute_score("econ_agent", json.dumps(base_action), "", extra_info)
        base_rewards.append(base_reward)
        
        # LoRA model
        lora_action, lora_response = generate_action(model, text, use_adapter=True)
        lora_reward = compute_score("econ_agent", json.dumps(lora_action), "", extra_info)
        lora_rewards.append(lora_reward)
        
        # Random action
        random_action = {"work": np.random.uniform(0, 1), "consumption": np.random.uniform(0, 1)}
        random_reward = compute_score("econ_agent", json.dumps(random_action), "", extra_info)
        random_rewards.append(random_reward)
        
        results.append({
            'timestep': extra_info.get('timestep', 'N/A'),
            'agent_id': extra_info.get('agent_id', 'N/A'),
            'buffer_ratio': extra_info.get('buffer_ratio', None),
            'r_okun_12m': extra_info.get('r_okun_12m', None),
            'r_phil_12m': extra_info.get('r_phil_12m', None),
            'u_mean_12m': extra_info.get('u_mean_12m', None),
            'base_work': base_action['work'],
            'base_consumption': base_action['consumption'],
            'base_reward': base_reward,
            'lora_work': lora_action['work'],
            'lora_consumption': lora_action['consumption'],
            'lora_reward': lora_reward,
            'random_reward': random_reward,
            'delta': lora_reward - base_reward,
        })
    
    results_df = pd.DataFrame(results)
    base_rewards = np.array(base_rewards)
    lora_rewards = np.array(lora_rewards)
    random_rewards = np.array(random_rewards)
    deltas = lora_rewards - base_rewards
    
    # ===== 4. 输出分析结果 =====
    print("\n" + "=" * 70)
    print("[4/6] 分析结果")
    print("=" * 70)
    
    # 1. KL (从训练日志看是 0，因为 use_kl_loss=false)
    print("\n1. KL 散度")
    print("-" * 40)
    print("  训练时 use_kl_loss=false, KL=0")
    print("  但 kl_coef=0.003 用于 reward shaping")
    
    # 2. Reward 分布
    print("\n2. Reward 分布")
    print("-" * 40)
    print(f"  {'模型':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print(f"  {'Base':<15} {base_rewards.mean():<10.4f} {base_rewards.std():<10.4f} {base_rewards.min():<10.4f} {base_rewards.max():<10.4f}")
    print(f"  {'LoRA (1000步)':<15} {lora_rewards.mean():<10.4f} {lora_rewards.std():<10.4f} {lora_rewards.min():<10.4f} {lora_rewards.max():<10.4f}")
    print(f"  {'Random':<15} {random_rewards.mean():<10.4f} {random_rewards.std():<10.4f} {random_rewards.min():<10.4f} {random_rewards.max():<10.4f}")
    print(f"\n  Delta (LoRA - Base): {deltas.mean():+.4f} ± {deltas.std():.4f}")
    
    # 3. Buffer ratio 与 work/consumption 关系
    print("\n3. Buffer Ratio 与 Work/Consumption 关系")
    print("-" * 40)
    br = results_df['buffer_ratio'].dropna()
    work = results_df['lora_work']
    cons = results_df['lora_consumption']
    
    if len(br) > 10:
        corr_work = np.corrcoef(br[:len(work)], work[:len(br)])[0, 1]
        corr_cons = np.corrcoef(br[:len(cons)], cons[:len(br)])[0, 1]
        print(f"  Buffer Ratio - Work 相关系数: {corr_work:.4f}")
        print(f"  Buffer Ratio - Consumption 相关系数: {corr_cons:.4f}")
        
        # 分组分析
        low_br = results_df[results_df['buffer_ratio'] < 2]
        high_br = results_df[results_df['buffer_ratio'] >= 2]
        if len(low_br) > 0 and len(high_br) > 0:
            print(f"\n  低 Buffer Ratio (<2): work={low_br['lora_work'].mean():.3f}, cons={low_br['lora_consumption'].mean():.3f}")
            print(f"  高 Buffer Ratio (>=2): work={high_br['lora_work'].mean():.3f}, cons={high_br['lora_consumption'].mean():.3f}")
    
    # 4. Work 分布
    print("\n4. Work 分布")
    print("-" * 40)
    print(f"  Base Model: mean={results_df['base_work'].mean():.3f}, std={results_df['base_work'].std():.3f}")
    print(f"  LoRA Model: mean={results_df['lora_work'].mean():.3f}, std={results_df['lora_work'].std():.3f}")
    print(f"\n  Work 分布 (LoRA):")
    print(f"    [0, 0.2]: {(results_df['lora_work'] < 0.2).sum()}")
    print(f"    [0.2, 0.4]: {((results_df['lora_work'] >= 0.2) & (results_df['lora_work'] < 0.4)).sum()}")
    print(f"    [0.4, 0.6]: {((results_df['lora_work'] >= 0.4) & (results_df['lora_work'] < 0.6)).sum()}")
    print(f"    [0.6, 0.8]: {((results_df['lora_work'] >= 0.6) & (results_df['lora_work'] < 0.8)).sum()}")
    print(f"    [0.8, 1.0]: {(results_df['lora_work'] >= 0.8).sum()}")
    
    # 5. 高 reward 样本的宏观关系检查
    print("\n5. 高 Reward 样本的宏观关系检查 (Top 20%)")
    print("-" * 40)
    top_20_threshold = np.percentile(lora_rewards, 80)
    top_samples = results_df[results_df['lora_reward'] >= top_20_threshold]
    
    if len(top_samples) > 0:
        r_okun = top_samples['r_okun_12m'].dropna()
        r_phil = top_samples['r_phil_12m'].dropna()
        u_mean = top_samples['u_mean_12m'].dropna()
        
        print(f"  Top 20% 样本数: {len(top_samples)}")
        if len(r_okun) > 0:
            print(f"\n  r_okun_12m (应为负值，目标 -0.894):")
            print(f"    mean={r_okun.mean():.3f}, std={r_okun.std():.3f}")
            print(f"    负值比例: {(r_okun < 0).sum()}/{len(r_okun)} = {100*(r_okun < 0).mean():.1f}%")
        
        if len(r_phil) > 0:
            print(f"\n  r_phil_12m (应为负值，目标 -0.676):")
            print(f"    mean={r_phil.mean():.3f}, std={r_phil.std():.3f}")
            print(f"    负值比例: {(r_phil < 0).sum()}/{len(r_phil)} = {100*(r_phil < 0).mean():.1f}%")
        
        if len(u_mean) > 0:
            print(f"\n  u_mean_12m (应在 4%-15%):")
            print(f"    mean={u_mean.mean():.3f}, std={u_mean.std():.3f}")
            in_range = ((u_mean >= 0.04) & (u_mean <= 0.15)).sum()
            print(f"    在合理区间比例: {in_range}/{len(u_mean)} = {100*in_range/len(u_mean):.1f}%")
    
    # 6. 与随机策略比较
    print("\n6. 与随机策略比较")
    print("-" * 40)
    lora_vs_random = lora_rewards - random_rewards
    print(f"  LoRA vs Random: {lora_vs_random.mean():+.4f} ± {lora_vs_random.std():.4f}")
    print(f"  LoRA 胜率: {(lora_vs_random > 0).sum()}/{len(lora_vs_random)} = {100*(lora_vs_random > 0).mean():.1f}%")
    
    # 胜率统计
    print("\n7. 胜率统计 (LoRA vs Base)")
    print("-" * 40)
    wins = (deltas > 0).sum()
    ties = (deltas == 0).sum()
    losses = (deltas < 0).sum()
    print(f"  LoRA 胜: {wins} ({100*wins/len(deltas):.1f}%)")
    print(f"  平局: {ties} ({100*ties/len(deltas):.1f}%)")
    print(f"  Base 胜: {losses} ({100*losses/len(deltas):.1f}%)")
    
    # ===== 结论 =====
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    if deltas.mean() > 0.02:
        print(f"✅ GRPO 训练有效! 平均提升: {deltas.mean():+.4f}")
    elif deltas.mean() > 0:
        print(f"⚠️ GRPO 效果较小. 平均 delta: {deltas.mean():+.4f}")
    else:
        print(f"❌ GRPO 可能无效. 平均下降: {deltas.mean():+.4f}")
    
    if (lora_vs_random > 0).mean() > 0.6:
        print(f"✅ 明显优于随机策略 (胜率 {100*(lora_vs_random > 0).mean():.1f}%)")
    else:
        print(f"⚠️ 相比随机策略提升不明显 (胜率 {100*(lora_vs_random > 0).mean():.1f}%)")
    
    # 保存结果
    results_df.to_csv('/workspace/QWEN2.5_42_7b_main/analysis_results_step1000.csv', index=False)
    print(f"\n详细结果已保存到: /workspace/QWEN2.5_42_7b_main/analysis_results_step1000.csv")
    
    return results_df

if __name__ == "__main__":
    main()
