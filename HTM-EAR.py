
"""
HTM-EAR: A Policy-Driven, Importance-Preserving Memory Substrate for Long-Running Agents
Experimental Framework  - FINAL PRODUCTION RELEASE
"""

import subprocess
import sys
import os
import time
import random
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm

# ============================================================
# DEPENDENCY INSTALLATION & REPRODUCIBILITY
# ============================================================
def install_deps():
    # Note: seaborn>=0.12 required for errorbar="sd" support
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "faiss-cpu", "sentence-transformers", "numpy", "pandas",
                           "tqdm", "matplotlib", "seaborn>=0.12"])

try:
    import faiss
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError:
    install_deps()
    import faiss
    from sentence_transformers import SentenceTransformer, CrossEncoder

# ============================================
# SECTION 1: SYSTEM CONFIGURATION
# ============================================

@dataclass
class Config:
    EMBEDDING_MODEL_NAME: str = "intfloat/e5-large-v2"
    CROSS_ENCODER_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    EMBEDDING_DIM: int = 1024

    # POLICY THRESHOLDS
    POLICY_ESSENTIAL_THRESHOLD: float = 0.85
    ROUTING_SIM_THRESHOLD: float = 0.84

    # SEARCH HYPERPARAMETERS
    WARM_K: int = 100
    COLD_K: int = 200

config = Config()

# ============================================
# SECTION 2: MODELS & HYBRID ROUTER
# ============================================

class InferenceSubstrate:
    def __init__(self, config: Config):
        print(f"Initializing HTM-EAR Inference Stack...")
        self.bi_encoder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self.cross_encoder = CrossEncoder(config.CROSS_ENCODER_NAME)

    def encode(self, text: str, is_query: bool = False) -> np.ndarray:
        prefix = "query: " if is_query else "passage: "
        return self.bi_encoder.encode(prefix + text, normalize_embeddings=True).astype("float32")

models = InferenceSubstrate(config)

class PolicyRouter:
    @staticmethod
    def extract_entities(text: str) -> set:
        patterns = [r'Node-\d+', r'ERR_[A-Z_]+', r'Subsystem-\d+', r'0x[0-9A-F]{4}']
        entities = set()
        for p in patterns:
            entities.update([m.lower() for m in re.findall(p, text)])
        return entities

router = PolicyRouter()

# ============================================
# SECTION 3: TIERED VECTOR STORAGE (L1 & L2)
# ============================================

@dataclass
class FactEntry:
    id: int
    text: str
    embedding: np.ndarray
    entities: set
    importance: float
    usage: int = 0

class StorageTier:
    def __init__(self, capacity: int, dim: int, name: str):
        self.name = name
        self.capacity = capacity
        self.index = faiss.IndexFlatIP(dim)
        self.items: Dict[int, FactEntry] = {}
        self.id_map: List[int] = []

    def add(self, item: FactEntry) -> List[FactEntry]:
        evicted = []
        if len(self.id_map) >= self.capacity:
            # Policy-Driven Eviction (Importance + Temporal Recency/Usage)
            scores = [(fid, 0.75 * self.items[fid].importance + 0.25 * min(self.items[fid].usage/10, 1.0))
                      for fid in self.id_map]
            scores.sort(key=lambda x: x[1])

            n_to_evict = max(1, int(0.15 * self.capacity))
            for fid, _ in scores[:n_to_evict]:
                evicted.append(self.items.pop(fid))
                self.id_map.remove(fid)

            self._rebuild_index()

        self.items[item.id] = item
        self.id_map.append(item.id)
        self.index.add(item.embedding.reshape(1, -1))
        return evicted

    def _rebuild_index(self):
        self.index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
        if self.id_map:
            self.index.add(np.vstack([self.items[fid].embedding for fid in self.id_map]))

    def search(self, q_emb: np.ndarray, k: int) -> List[Tuple[FactEntry, float]]:
        if not self.id_map: return []
        scores, indices = self.index.search(q_emb.reshape(1, -1), min(k, len(self.id_map)))
        results = []
        for s, idx in zip(scores[0], indices[0]):
            if idx != -1:
                fid = self.id_map[idx]
                results.append((self.items[fid], float(s)))
        return results

# ============================================
# SECTION 4: THE HTM-EAR ENGINE
# ============================================

class HTMEngine:
    def __init__(self, l1_cap: int, l2_cap: int, mode: str = "full"):
        self.mode = mode
        self.L1 = StorageTier(l1_cap, config.EMBEDDING_DIM, "L1_Working")
        self.L2 = StorageTier(l2_cap, config.EMBEDDING_DIM, "L2_Archive")
        self.stats = {"essential_lost": 0, "pruned_total": 0, "latency": []}

    def ingest(self, fact_id: int, text: str, importance: float):
        emb = models.encode(text)
        item = FactEntry(fact_id, text, emb, router.extract_entities(text), importance)

        if self.mode == "flat":
            self.L2.capacity = 999999
            self.L2.add(item)
            return

        evicted_from_l1 = self.L1.add(item)
        for e in evicted_from_l1:
            permanently_deleted = self.L2.add(e)
            self.stats["pruned_total"] += len(permanently_deleted)
            for d in permanently_deleted:
                if d.importance >= config.POLICY_ESSENTIAL_THRESHOLD:
                    self.stats["essential_lost"] += 1

    def retrieve(self, query: str, k: int = 5) -> List[int]:
        t0 = time.perf_counter()
        q_emb = models.encode(query, is_query=True)
        q_ents = router.extract_entities(query)

        if self.mode == "flat":
            candidates = self.L2.search(q_emb, config.COLD_K)
        else:
            candidates = self.L1.search(q_emb, config.WARM_K)
            if self.mode != "no_gate":
                if not candidates or candidates[0][1] < config.ROUTING_SIM_THRESHOLD or not q_ents.issubset(candidates[0][0].entities):
                    candidates = self.L2.search(q_emb, config.COLD_K)

        if not candidates:
            self.stats["latency"].append(time.perf_counter() - t0)
            return []

        scored = []
        for item, sim in candidates:
            overlap = len(q_ents.intersection(item.entities))
            score = (sim**3) + (0.8 * overlap) + (0.1 * item.importance)
            scored.append((item, score))
        scored.sort(key=lambda x: x[1], reverse=True)

        if self.mode != "no_ce":
            top_set = scored[:20]
            ce_scores = models.cross_encoder.predict([[query, x[0].text] for x in top_set])
            final = sorted([(top_set[i][0], (0.85 * ce_scores[i]) + (0.15 * top_set[i][1])) for i in range(len(top_set))], key=lambda x: x[1], reverse=True)
        else:
            final = [(x[0], x[1]) for x in scored]

        self.stats["latency"].append(time.perf_counter() - t0)
        if final: final[0][0].usage += 1
        return [f[0].id for f in final[:k]]

# ============================================
# SECTION 5: DATASET GENERATION
# ============================================

def generate_dataset(n, seed):
    random.seed(seed)
    np.random.seed(seed)
    facts, queries = {}, {}
    regions = [f"Region-{r}" for r in range(25)]
    causes = ["Memory Overflow", "CPU Throttling", "Disk Saturation", "Kernel Panic", "I/O Wait", "OOM Killer"]
    services = ["Auth-Svc", "Inference-Engine", "Data-Lake", "API-Gateway", "Stream-Processor"]

    for i in range(1, n + 1):
        reg = random.choice(regions)
        cause = random.choice(causes)
        svc = random.choice(services)
        sub = f"Subsystem-{i % 1000}"
        status = f"0x{i:04X}"
        is_essential = i % 10 == 0 or "Panic" in cause
        importance = 0.95 if is_essential else 0.5
        facts[i] = {"text": f"ENTRY {i}: {svc} in {reg} logged {cause} at {sub}. Hex: {status}.", "importance": importance}
        queries[i] = f"Retrieve the Hex Code for the {cause} event in {reg} involving {svc} at {sub}."
    return facts, queries

# ============================================
# SECTION 6: THE STATISTICAL SUITE
# ============================================

def run_publication_benchmark(num_seeds=5):
    scenarios = [
        {"name": "Scenario A: Overflow", "n": 3000, "l1": 500, "l2": 5000},
        {"name": "Scenario B: Saturation (Forgetting)", "n": 15000, "l1": 500, "l2": 5000},
        {"name": "Scenario C: Control (High-Cap)", "n": 5000, "l1": 1000, "l2": 14000}
    ]
    modes = ["full", "flat", "no_ce", "no_gate"]
    all_runs = []

    print(f"Executing high-fidelity evaluation across {num_seeds} seeds...")
    print("Note: Running with E5-Large and Cross-Encoder; execution may take significant time on CPU.")

    for sc in scenarios:
        print(f"\n[SCENARIO: {sc['name']}]")
        for mode in modes:
            print(f"  > Mode: {mode.upper()}")
            for s in tqdm(range(num_seeds), desc="  Seeds", leave=False):
                current_seed = 42 + s
                facts, queries = generate_dataset(sc['n'], current_seed)
                engine = HTMEngine(sc['l1'], sc['l2'], mode=mode)

                # 1. Ingestion Phase
                for fid, d in facts.items():
                    engine.ingest(fid, d['text'], d['importance'])

                # 2. Evaluation Phase (Decoupled Latency per Phase)
                for label, ids in [("Active", range(sc['n']-100, sc['n']+1)), ("History", range(1, 101))]:
                    engine.stats["latency"] = []
                    mrr_vals = []
                    for sid in ids:
                        res = engine.retrieve(queries[sid])
                        mrr_vals.append(1 / (res.index(sid) + 1) if sid in res else 0)

                    # Compute latency with safety fallback to NaN to preserve statistical validity
                    lat = (np.mean(engine.stats["latency"]) * 1000) if engine.stats["latency"] else np.nan

                    all_runs.append({
                        "Scenario": sc['name'],
                        "Mode": mode,
                        "Temporal": label,
                        "Seed": current_seed,
                        "MRR": np.mean(mrr_vals),
                        "Latency_ms": lat,
                        "Essential_Lost": engine.stats["essential_lost"],
                        "Pruned_Total": engine.stats["pruned_total"]
                    })

    return pd.DataFrame(all_runs)

def main():
    # Benchmark execution
    results_df = run_publication_benchmark(num_seeds=5)

    print("\n" + "="*95)
    print("HTM-EAR: MULTI-SEED ROBUSTNESS ANALYSIS (N=15,000 | FINAL PRODUCTION SUITE)")
    print("="*95)

    # Filter for Scenario B for detailed metrics
    scen_b_mask = results_df["Scenario"].str.contains("Scenario B")

    # 1. TABLE 1: Retrieval Performance (Mean +/- Std)
    # Correctly handles MRR per phase condition
    table_b_mrr = results_df[scen_b_mask].groupby(["Mode", "Temporal"])["MRR"].agg(['mean', 'std']).unstack()
    print("\n[TABLE 1: RETRIEVAL PERFORMANCE (MEAN +/- STD)]")
    print(table_b_mrr.to_string())

    # 2. TABLE 2: System Robustness Metrics (Mean +/- Std)
    # FIX: Using only 'Active' slice to avoid duplicating Pruned/Essential counts in seed outcome calculation
    sys_stats = (results_df[scen_b_mask & (results_df["Temporal"] == "Active")]
                .groupby("Mode")[["Latency_ms", "Essential_Lost", "Pruned_Total"]]
                .agg(['mean', 'std']))

    print("\n[TABLE 2: SYSTEM ROBUSTNESS METRICS (MEAN +/- STD)]")
    print("Note: Latency corresponds to the Active retrieval phase.")
    print(sys_stats.to_string())

    # Visual 1: Precision Decay (Correct Error Bar Mapping via Seaborn SD)
    plt.figure(figsize=(14, 7))
    sns.set_style("whitegrid")
    sns.barplot(data=results_df[scen_b_mask], x="Mode", y="MRR", hue="Temporal",
                palette="coolwarm", errorbar="sd", capsize=.1)

    plt.title("Ablation Study: Precision Decay in Saturated Systems (Scenario B)", fontsize=14, fontweight='bold')
    plt.ylabel("Mean Reciprocal Rank (MRR)", fontsize=13)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig("precision_decay_v8.4.png")

    # Visual 2: Pareto Chart
    # FIX: Explicit numeric selection for pandas version stability + Active phase filtering
    plt.figure(figsize=(10, 6))
    pareto_data = (results_df[scen_b_mask & (results_df["Temporal"] == "Active")]
                  .groupby("Mode")[["Latency_ms", "MRR"]]
                  .mean()
                  .reset_index())

    sns.scatterplot(data=pareto_data, x="Latency_ms", y="MRR", hue="Mode", s=300,
                    palette="Set1", edgecolor="black")

    for _, row in pareto_data.iterrows():
        plt.annotate(row['Mode'].upper(), (row['Latency_ms'], row['MRR']),
                     textcoords="offset points", xytext=(0,12), ha='center', fontweight='bold')

    plt.title("Pareto Frontier: Active Retrieval Performance vs. Phase Latency", fontsize=16, fontweight='bold')
    plt.axhspan(0, 0.6, color='red', alpha=0.1, label="Failure Zone")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pareto_frontier_v8.4.png")

    plt.show()

if __name__ == "__main__":
    main()