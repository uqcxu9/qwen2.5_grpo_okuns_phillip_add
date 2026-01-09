"""
验证训练后的模型（使用 LoRA adapter，未 merge）
- 加载 base model + LoRA adapter
- 在验证集上生成 action
- 用 reward.py 评估
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
LORA_PATH = "/workspace/QWEN2.5_42_7b_main/checkpoints/global_step_350/actor/lora_adapter"
VAL_DATA_PATH = "/workspace/QWEN2.5_42_7b_main/RL/verl_dataset/val.parquet"
NUM_SAMPLES = 200  # 验证样本数（与 prepare_verl_data.py 一致）

def extract_action(response_text):
    """从模型输出中提取 action"""
    try:
        # 尝试直接解析 JSON
        response_text = response_text.strip()
        
        # 查找 JSON 块
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        
        # 查找 { } 包围的部分
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start != -1 and end > start:
            json_str = response_text[start:end]
            action = json.loads(json_str)
            
            # 标准化字段名
            work = action.get('work', action.get('work_decision', action.get('Work', 0.5)))
            consumption = action.get('consumption', action.get('consumption_prop', action.get('Consumption', 0.5)))
            
            return {
                "work": float(work),
                "consumption": float(consumption)
            }
    except Exception as e:
        pass
    
    # 默认值
    return {"work": 0.5, "consumption": 0.5}

def main():
    print("=" * 60)
    print("验证训练后模型（LoRA adapter，未 merge）")
    print("=" * 60)
    
    # ===== 1. 加载模型 =====
    print("\n[1/4] 加载模型...")
    print(f"  Base model: {BASE_MODEL_PATH}")
    print(f"  LoRA adapter: {LORA_PATH}")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载 base model + LoRA adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"  # 避免 flash attention 问题
    )
    
    # 加载 LoRA adapter（会 in-place 修改 base_model）
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()
    print("  模型加载完成! (可通过 disable_adapter 切换 base/lora)")
    
    # ===== 2. 加载验证集 =====
    print("\n[2/4] 加载验证集...")
    val_df = pd.read_parquet(VAL_DATA_PATH)
    print(f"  验证集总量: {len(val_df)} 条")
    
    # 抽样
    if len(val_df) > NUM_SAMPLES:
        val_df = val_df.sample(n=NUM_SAMPLES, random_state=42)
    print(f"  本次验证: {len(val_df)} 条")
    
    # ===== 3. 生成并评估 =====
    print("\n[3/4] 生成 action 并评估...")
    
    results = []
    base_rewards = []
    lora_rewards = []
    
    def generate_action(model, text):
        """用模型生成 action"""
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        action = extract_action(response)
        return action, response
    
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df)):
        # 正确处理 prompt（它是 list of messages）
        prompt_messages = row["prompt"]
        
        # 正确处理 extra_info
        extra_info = row.get("extra_info", {})
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except:
                extra_info = {}
        elif extra_info is None:
            extra_info = {}
        
        # prompt_messages 本来就是 list[{"role":...,"content":...}]
        # 直接喂给 chat_template
        text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Base model 生成 (禁用 LoRA adapter)
        with model.disable_adapter():
            base_action, base_response = generate_action(model, text)
        base_reward = compute_score(
            data_source="econ_agent",
            solution_str=json.dumps(base_action),
            ground_truth="",
            extra_info=extra_info
        )
        base_rewards.append(base_reward)
        
        # LoRA model 生成 (启用 LoRA adapter)
        lora_action, lora_response = generate_action(model, text)
        lora_reward = compute_score(
            data_source="econ_agent",
            solution_str=json.dumps(lora_action),
            ground_truth="",
            extra_info=extra_info
        )
        lora_rewards.append(lora_reward)
        
        results.append({
            'timestep': extra_info.get('timestep', 'N/A'),
            'agent_id': extra_info.get('agent_id', 'N/A'),
            'base_work': base_action['work'],
            'base_consumption': base_action['consumption'],
            'base_reward': base_reward,
            'lora_work': lora_action['work'],
            'lora_consumption': lora_action['consumption'],
            'lora_reward': lora_reward,
            'delta': lora_reward - base_reward,
            'base_response': base_response[:150],
            'lora_response': lora_response[:150]
        })
    
    # ===== 4. 统计结果 =====
    print("\n[4/4] 统计结果...")
    base_rewards = np.array(base_rewards)
    lora_rewards = np.array(lora_rewards)
    deltas = lora_rewards - base_rewards
    
    print("\n" + "=" * 60)
    print("验证结果: Base Model vs LoRA Model")
    print("=" * 60)
    print(f"样本数: {len(base_rewards)}")
    
    print(f"\n{'指标':<20} {'Base Model':<15} {'LoRA Model':<15} {'Delta':<15}")
    print("-" * 65)
    print(f"{'平均 reward':<20} {base_rewards.mean():<15.4f} {lora_rewards.mean():<15.4f} {deltas.mean():<+15.4f}")
    print(f"{'标准差':<20} {base_rewards.std():<15.4f} {lora_rewards.std():<15.4f} {deltas.std():<15.4f}")
    print(f"{'最小值':<20} {base_rewards.min():<15.4f} {lora_rewards.min():<15.4f} {deltas.min():<+15.4f}")
    print(f"{'最大值':<20} {base_rewards.max():<15.4f} {lora_rewards.max():<15.4f} {deltas.max():<+15.4f}")
    print(f"{'中位数':<20} {np.median(base_rewards):<15.4f} {np.median(lora_rewards):<15.4f} {np.median(deltas):<+15.4f}")
    
    # 胜率统计
    wins = (deltas > 0).sum()
    ties = (deltas == 0).sum()
    losses = (deltas < 0).sum()
    print(f"\n胜率统计 (LoRA vs Base):")
    print(f"  LoRA 胜: {wins} ({100*wins/len(deltas):.1f}%)")
    print(f"  平局: {ties} ({100*ties/len(deltas):.1f}%)")
    print(f"  Base 胜: {losses} ({100*losses/len(deltas):.1f}%)")
    
    # Reward 分布对比
    print(f"\nReward 分布:")
    print(f"{'区间':<15} {'Base':<12} {'LoRA':<12}")
    print("-" * 39)
    for label, cond_base, cond_lora in [
        ("< 0", base_rewards < 0, lora_rewards < 0),
        ("0 ~ 0.2", (base_rewards >= 0) & (base_rewards < 0.2), (lora_rewards >= 0) & (lora_rewards < 0.2)),
        ("0.2 ~ 0.5", (base_rewards >= 0.2) & (base_rewards < 0.5), (lora_rewards >= 0.2) & (lora_rewards < 0.5)),
        (">= 0.5", base_rewards >= 0.5, lora_rewards >= 0.5),
    ]:
        print(f"{label:<15} {cond_base.sum():<12} {cond_lora.sum():<12}")
    
    # 保存详细结果
    results_df = pd.DataFrame(results)
    results_df.to_csv('/workspace/QWEN2.5_42_7b_main/validation_results.csv', index=False)
    print(f"\n详细结果已保存到: /workspace/QWEN2.5_42_7b_main/validation_results.csv")
    
    # 打印几个样例
    print("\n样例输出:")
    print("-" * 60)
    for i, r in enumerate(results[:3]):
        print(f"[样例 {i+1}] timestep={r['timestep']}, agent={r['agent_id']}")
        print(f"  Base:  work={r['base_work']:.2f}, cons={r['base_consumption']:.2f}, reward={r['base_reward']:.4f}")
        print(f"  LoRA:  work={r['lora_work']:.2f}, cons={r['lora_consumption']:.2f}, reward={r['lora_reward']:.4f}")
        print(f"  Delta: {r['delta']:+.4f}")
        print()
    
    # 结论
    print("=" * 60)
    if deltas.mean() > 0.01:
        print(f"✅ GRPO 训练有效! 平均提升: {deltas.mean():+.4f}")
    elif deltas.mean() > -0.01:
        print(f"⚠️ GRPO 效果不明显. 平均 delta: {deltas.mean():+.4f}")
    else:
        print(f"❌ GRPO 可能有问题. 平均下降: {deltas.mean():+.4f}")
    print("=" * 60)
    
    return deltas.mean()

if __name__ == "__main__":
    main()
