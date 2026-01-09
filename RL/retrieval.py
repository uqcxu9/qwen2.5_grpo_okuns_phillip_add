import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Optional, Tuple


class GoodDecisionMemory:
    """
    Good decision memory store v3
    
    Changes:
    - Removed employed/unemployed bucketing (was semantic mismatch)
    - Single unified index with post-retrieval balancing
    - Better handling of work_decision diversity
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize memory store
        
        Args:
            csv_path: path to good_decisions.csv
        """
        self.df = pd.read_csv(csv_path)
        
        # Core features (required)
        self.core_feature_cols = [
            'curr_income',
            'curr_wealth',
            'curr_dpi',
            'prev_consumption',
        ]
        
        # Optional features (use if available)
        self.optional_feature_cols = [
            'prev_income',
            'prev_dpi',
        ]
        
        # Determine actual features to use
        self.feature_cols = []
        for col in self.core_feature_cols:
            if col in self.df.columns:
                self.feature_cols.append(col)
            else:
                raise ValueError(f"good_decisions.csv missing required column: {col}")
        
        for col in self.optional_feature_cols:
            if col in self.df.columns:
                self.feature_cols.append(col)
        
        print(f"[GoodDecisionMemory] Retrieval features: {self.feature_cols}")
        print(f"[GoodDecisionMemory] Total samples: {len(self.df)}")
        
        # Count work_decision distribution for info
        if 'work_decision' in self.df.columns:
            work_1 = (self.df['work_decision'] == 1.0).sum()
            work_0 = (self.df['work_decision'] == 0.0).sum()
            print(f"[GoodDecisionMemory] work_decision=1: {work_1}, work_decision=0: {work_0}")
        
        # Build single unified index (no bucketing)
        self._build_index()
    
    def _build_index(self):
        """Build single KNN index on all data (no bucketing)"""
        
        all_features = self.df[self.feature_cols].fillna(0).values
        self.scaler = StandardScaler()
        self.scaler.fit(all_features)
        
        self.features_scaled = self.scaler.transform(all_features)
        
        n_neighbors = min(50, len(self.df))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        self.nn.fit(self.features_scaled)
    
    def retrieve(
        self, 
        current_state: Dict, 
        k: int = 3,
        score_threshold: float = 10.0,
        balance_job: bool = True  # 改名
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Retrieve k most similar good decisions
        
        Args:
            current_state: current state dictionary
            k: number of samples to return
            score_threshold: minimum score threshold
            balance_work: if True, try to include both work=1 and work=0 samples
        
        Returns:
            (retrieved samples DataFrame, distance array)
        """
        # Build query vector
        query_values = []
        for col in self.feature_cols:
            val = self._get_feature_value(current_state, col)
            query_values.append(float(val) if val is not None else 0.0)
        
        query = np.array([query_values])
        query_scaled = self.scaler.transform(query)
        
        # Step 1: Get k*10 nearest candidates (more candidates for balancing)
        n_candidates = min(k * 10, len(self.df))
        distances, indices = self.nn.kneighbors(query_scaled, n_neighbors=n_candidates)
        
        # Step 2: Get candidate samples
        candidates = self.df.iloc[indices[0]].copy()
        candidates['_distance'] = distances[0]
        
        # Step 3: Filter score >= threshold
        if score_threshold > 0 and 'score' in candidates.columns:
            filtered = candidates[candidates['score'] >= score_threshold]
            if len(filtered) >= k:
                candidates = filtered
        
        # Step 4: Sort by score desc, distance asc
        if 'score' in candidates.columns:
            candidates = candidates.sort_values(
                by=['score', '_distance'], 
                ascending=[False, True]
            )
        else:
            candidates = candidates.sort_values(by='_distance', ascending=True)
        
    # Step 5: Balance job_status if requested
        if balance_job and 'job_status' in candidates.columns and k >= 2:
            results = self._balance_work_selection(candidates, k)
        else:
            results = candidates.head(k).copy()
        
        result_distances = results['_distance'].values
        results = results.drop(columns=['_distance'], errors='ignore')
        
        return results, result_distances
    
    def _balance_work_selection(self, candidates: pd.DataFrame, k: int) -> pd.DataFrame:
        """
        Select k samples with job_status diversity (not work_decision)
        Try to include at least one employed and one unemployed if available
        """
        # 改用 job_status 平衡（0=失业，1=就业）
        if 'job_status' not in candidates.columns:
            # fallback: 直接取 top k
            return candidates.head(k)
        
        employed = candidates[candidates['job_status'] == 1]
        unemployed = candidates[candidates['job_status'] == 0]
        
        selected = []
        
        # Ensure at least one from each category if available
        if len(employed) > 0 and len(unemployed) > 0:
            selected.append(employed.head(1))
            selected.append(unemployed.head(1))
            remaining = k - 2
            if remaining > 0:
                already_idx = set(employed.head(1).index) | set(unemployed.head(1).index)
                rest = candidates[~candidates.index.isin(already_idx)].head(remaining)
                selected.append(rest)
        else:
            selected.append(candidates.head(k))
        
        return pd.concat(selected).head(k)
    
    def _get_feature_value(self, state: Dict, col: str) -> float:
        """Get feature value from state, compatible with different key names"""
        if col in state:
            return state[col]
        
        aliases = {
            'curr_income': ['income'],
            'curr_wealth': ['wealth'],
            'curr_dpi': ['dpi'],
            'prev_consumption': ['consumption', 'last_consumption'],
            'prev_income': ['last_income'],
            'prev_dpi': ['last_dpi'],
        }
        
        for alias in aliases.get(col, []):
            if alias in state:
                return state[alias]
        
        return 0.0


def format_few_shot_examples(
    examples: pd.DataFrame,
    include_score: bool = False,
    max_examples: int = 3
) -> str:
    """
    Format retrieved samples as few-shot prompt text
    Shows JSON format with soft work values to align with output format
    """
    if len(examples) == 0:
        return ""
    
    lines = ["[Reference Examples]"]
    
    for idx, (_, row) in enumerate(examples.head(max_examples).iterrows()):
        # Income change (optional)
        income_change_str = ""
        if 'prev_income' in row and pd.notna(row['prev_income']):
            income_change = row['curr_income'] - row['prev_income']
            income_change_str = f", income_change={income_change:+.0f}"
        
        # ===== 核心修正：按 job_status 处理 work =====
        job_status = row.get('job_status', 1)  # 0=失业, 1=就业
        
        if job_status == 0:
            # 失业样本：明确是“无劳动决策”
            w = 0.0
        else:
            work_decision = row.get('work_decision', 1.0)
            try:
                w = float(work_decision)
            except:
                w = 1.0
            w = float(np.clip(w, 0.0, 1.0))
        
        # ===== 软化（避免模型死学 0 / 1）=====
        if w >= 0.5:
            soft_work = 0.8 + 0.2 * (w - 0.5) / 0.5   # [0.8, 1.0]
        else:
            soft_work = 0.0 + 0.2 * (w / 0.5)         # [0.0, 0.2]
        
        consumption = row['consumption_prop']
        
        example_text = (
            f"Example {idx + 1}: income={row['curr_income']:.0f}, "
            f"wealth={row['curr_wealth']:.0f}, DPI={row['curr_dpi']:.0f}"
            f"{income_change_str}\n"
            f"  -> {{\"work\": {soft_work:.1f}, \"consumption\": {consumption:.2f}}}"
        )
        
        if include_score and 'score' in row:
            example_text += f" (score={row['score']:.0f})"
        
        lines.append(example_text)
    
    return "\n".join(lines)



def create_cheatsheet(
    examples: pd.DataFrame,
    current_state: Optional[Dict] = None
) -> str:
    """
    DC-style Curator: compress retrieved samples into concise decision rules
    """
    if len(examples) == 0:
        return ""
    
    rules = ["[Decision Reference Rules]"]
    
    # 1. Consumption ratio statistics
    avg_prop = examples['consumption_prop'].mean()
    std_prop = examples['consumption_prop'].std() if len(examples) > 1 else 0.05
    min_prop = max(0, avg_prop - std_prop)
    max_prop = min(1, avg_prop + std_prop)
    
    rules.append(f"* Suggested consumption: {min_prop:.2f} ~ {max_prop:.2f} (typical: {avg_prop:.2f})")
    
    # 2. MPC statistics (if available)
    if 'yearly_mpc' in examples.columns:
        valid_mpc = examples['yearly_mpc'].dropna()
        if len(valid_mpc) > 0:
            avg_mpc = valid_mpc.mean()
            rules.append(f"* Marginal Propensity to Consume (MPC): {avg_mpc:.2f}")
    
    # 3. Work decision statistics (soft guidance)
    if 'work_decision' in examples.columns:
        work_rate = examples['work_decision'].mean()
        if work_rate > 0.7:
            rules.append(f"* Work tendency: high (suggest work >= 0.7)")
        elif work_rate < 0.3:
            rules.append(f"* Work tendency: low (suggest work <= 0.3)")
        else:
            rules.append(f"* Work tendency: mixed (consider your situation)")
    
    # 4. Suggestions based on current state
    if current_state:
        curr_dpi = current_state.get('dpi', current_state.get('curr_dpi', 0))
        avg_dpi = examples['curr_dpi'].mean()
        
        if curr_dpi > avg_dpi * 1.2:
            rules.append(f"* Current DPI is high, consider higher consumption")
        elif curr_dpi < avg_dpi * 0.8:
            rules.append(f"* Current DPI is low, consider conservative consumption")
    
    return "\n".join(rules)


def build_prompt_with_memory(
    current_state: Dict,
    memory: GoodDecisionMemory,
    k: int = 3,
    use_cheatsheet: bool = True,
    include_examples: bool = True
) -> str:
    """
    Build prompt with memory retrieval (DC-RS style)
    """
    # 1. Retrieval
    examples, distances = memory.retrieve(current_state, k=k)
    
    # 2. Curator
    memory_section = ""
    
    if use_cheatsheet and len(examples) > 0:
        cheatsheet = create_cheatsheet(examples, current_state)
        memory_section += cheatsheet + "\n\n"
    
    if include_examples and len(examples) > 0:
        few_shot_text = format_few_shot_examples(examples, include_score=False)
        memory_section += few_shot_text + "\n\n"
    
    # 3. Current state
    income = current_state.get('income', current_state.get('curr_income', 0))
    wealth = current_state.get('wealth', current_state.get('curr_wealth', 0))
    dpi = current_state.get('dpi', current_state.get('curr_dpi', 0))
    prev_c = current_state.get('prev_consumption', current_state.get('consumption', 0))
    
    state_section = f"""[Current State]
* Income: {income:.0f}
* Wealth: {wealth:.0f}
* Disposable Income (DPI): {dpi:.0f}
* Previous Consumption: {prev_c:.0f}"""
    
    # Forced output format
    format_instruction = """
[Output Requirement]
Output only one line of JSON, no explanation:
{"work": 0.xx, "consumption": 0.xx}

Both work and consumption are decimals between 0 and 1."""
    
    prompt = f"""{memory_section}{state_section}
{format_instruction}"""
    
    return prompt


def get_memory_context_for_prompt(
    current_state: Dict,
    memory: GoodDecisionMemory,
    k: int = 3
) -> str:
    """
    Return only the memory section (for appending to existing prompt)
    """
    examples, _ = memory.retrieve(current_state, k=k)
    
    if len(examples) == 0:
        return ""
    
    cheatsheet = create_cheatsheet(examples, current_state)
    few_shot = format_few_shot_examples(examples, include_score=False)
    
    return f"{cheatsheet}\n\n{few_shot}"


# ==================== Test Code ====================
if __name__ == "__main__":
    import os
    
    csv_path = "good_decisions.csv"
    
    if not os.path.exists(csv_path):
        print(f"Test file not found: {csv_path}")
        print("Please provide good_decisions.csv path")
    else:
        memory = GoodDecisionMemory(csv_path)
        
        test_state = {
            'income': 5000,
            'wealth': 10000,
            'dpi': 4500,
            'prev_consumption': 3500,
        }
        
        print("\n" + "="*50)
        print("Test retrieval (unified index, balanced selection)")
        print("="*50)
        
        examples, distances = memory.retrieve(test_state, k=3)
        print(f"\nRetrieved {len(examples)} samples:")
        if len(examples) > 0:
            cols = ['curr_income', 'curr_wealth', 'curr_dpi', 'work_decision', 'consumption_prop', 'score']
            cols = [c for c in cols if c in examples.columns]
            print(examples[cols].to_string())
        
        print("\n" + "="*50)
        print("Full prompt (with JSON format examples)")
        print("="*50)
        full_prompt = build_prompt_with_memory(test_state, memory, k=3)
        print(full_prompt)
        
        print("\n" + "="*50)
        print("Memory context only")
        print("="*50)
        context = get_memory_context_for_prompt(test_state, memory, k=3)
        print(context)
