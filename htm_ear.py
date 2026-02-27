
"""
HTM-EAR: A Policy-Driven, Importance-Preserving Memory Substrate for Long-Running Agents
Experimental Framework
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
import urllib.request
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm

# ============================================================
# DEPENDENCY INSTALLATION & REPRODUCIBILITY
# ============================================================
def install_deps():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "hnswlib", "sentence-transformers", "numpy", "pandas",
                           "tqdm", "matplotlib", "seaborn>=0.12"])

try:
    import hnswlib
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError:
    install_deps()
    import hnswlib
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
    def extract_entities(text: str, patterns: List[str] = None) -> set:
        if patterns is None:
            # Default patterns for synthetic dataset
            patterns = [r'Node-\d+', r'ERR_[A-Z_]+', r'Subsystem-\d+', r'0x[0-9A-F]{4}']
        entities = set()
        for p in patterns:
            entities.update([m.lower() for m in re.findall(p, text)])
        return entities

router = PolicyRouter()

# ============================================
# SECTION 3: TIERED VECTOR STORAGE (L1 & L2) with HNSWlib (safe max_elements + slot reuse)
# ============================================

@dataclass
class FactEntry:
    id: int
    text: str
    embedding: np.ndarray
    entities: set
    importance: float
    usage: int = 0
    last_access: float = field(default_factory=time.time)  # for LRU

class StorageTier:
    def __init__(self, capacity: int, dim: int, name: str, max_historical_inserts: int = 100000):
        """
        capacity: logical capacity (triggers eviction)
        max_historical_inserts: physical allocation for HNSWlib (must be >= total inserts over lifetime)
        """
        self.name = name
        self.capacity = capacity
        # Enable slot reuse with allow_replace_deleted=True
        self.index = hnswlib.Index(space='ip', dim=dim)
        self.index.init_index(max_elements=max_historical_inserts, ef_construction=200, M=16, allow_replace_deleted=True)
        self.index.set_ef(50)  # recall/speed tradeoff
        self.items: Dict[int, FactEntry] = {}
        self.id_map: List[int] = []

    def add(self, item: FactEntry, policy: str = "policy") -> List[FactEntry]:
        """
        Add item to the tier. If capacity is reached, evict items according to policy.
        Returns list of evicted FactEntry objects.
        """
        evicted = []
        if len(self.id_map) >= self.capacity:
            # Compute scores based on policy
            if policy == "lru":
                scores = [(fid, self.items[fid].last_access) for fid in self.id_map]
            else:  # default policy: importance + usage
                scores = [(fid, 0.75 * self.items[fid].importance + 0.25 * min(self.items[fid].usage/10, 1.0))
                          for fid in self.id_map]

            # Sort ascending (lowest score evicted first)
            scores.sort(key=lambda x: x[1])

            n_to_evict = max(1, int(0.15 * self.capacity))
            for fid, _ in scores[:n_to_evict]:
                self.index.mark_deleted(fid)
                evicted.append(self.items.pop(fid))
                self.id_map.remove(fid)

        # Add new item (reuse deleted slots with replace_deleted=True)
        self.items[item.id] = item
        self.id_map.append(item.id)
        self.index.add_items(item.embedding.reshape(1, -1), np.array([item.id]), replace_deleted=True)
        return evicted

    def search(self, q_emb: np.ndarray, k: int) -> List[Tuple[FactEntry, float]]:
        if not self.id_map:
            return []
        labels, distances = self.index.knn_query(q_emb.reshape(1, -1), k=min(k, len(self.id_map)))
        results = []
        for label, dist in zip(labels[0], distances[0]):
            if label != -1 and label in self.items:
                # Convert HNSW distance (1 - IP) back to similarity (IP)
                sim = 1.0 - float(dist)
                results.append((self.items[label], sim))
        return results

# ============================================
# SECTION 4: THE HTM-EAR ENGINE (with LRU mode and safe HNSW allocation)
# ============================================

class HTMEngine:
    def __init__(self, l1_cap: int, l2_cap: int, mode: str = "full", total_facts_estimate: int = 20000):
        """
        total_facts_estimate: upper bound on number of facts that will ever be inserted (for HNSW physical capacity)
        """
        self.mode = mode

        if mode == "oracle_unbounded":
            # Oracle (unbounded) mode: single store with huge capacity, no eviction
            self.L1 = StorageTier(1, config.EMBEDDING_DIM, "L1_Dummy", max_historical_inserts=1)
            self.L2 = StorageTier(total_facts_estimate, config.EMBEDDING_DIM, "L2_Oracle", max_historical_inserts=total_facts_estimate + 1000)
        else:
            # Normal tiered modes
            l1_max = max(total_facts_estimate, l1_cap * 2)
            l2_max = max(total_facts_estimate, l2_cap * 2)
            self.L1 = StorageTier(l1_cap, config.EMBEDDING_DIM, "L1_Working", max_historical_inserts=l1_max)
            self.L2 = StorageTier(l2_cap, config.EMBEDDING_DIM, "L2_Archive", max_historical_inserts=l2_max)

        self.stats = {"essential_lost": 0, "pruned_total": 0, "latency": []}

    def ingest(self, fact_id: int, text: str, importance: float):
        emb = models.encode(text)
        item = FactEntry(fact_id, text, emb, router.extract_entities(text), importance)

        if self.mode == "oracle_unbounded":
            # Oracle: add directly to L2 (huge capacity, no eviction)
            self.L2.add(item)
            return

        if self.mode == "lru":
            # LRU mode: both tiers use LRU eviction
            evicted_from_l1 = self.L1.add(item, policy="lru")
            for e in evicted_from_l1:
                permanently_deleted = self.L2.add(e, policy="lru")
                self.stats["pruned_total"] += len(permanently_deleted)
                for d in permanently_deleted:
                    if d.importance >= config.POLICY_ESSENTIAL_THRESHOLD:
                        self.stats["essential_lost"] += 1
            return

        # Default (full, no_ce, no_gate): policy eviction in L1, policy eviction in L2
        evicted_from_l1 = self.L1.add(item)  # default policy
        for e in evicted_from_l1:
            permanently_deleted = self.L2.add(e)  # default policy
            self.stats["pruned_total"] += len(permanently_deleted)
            for d in permanently_deleted:
                if d.importance >= config.POLICY_ESSENTIAL_THRESHOLD:
                    self.stats["essential_lost"] += 1

    def retrieve(self, query: str, k: int = 5) -> List[int]:
        t0 = time.perf_counter()
        q_emb = models.encode(query, is_query=True)
        q_ents = router.extract_entities(query)

        if self.mode == "oracle_unbounded":
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

        if self.mode != "no_ce" and self.mode != "lru":  # LRU also skips cross-encoder for speed (optional)
            top_set = scored[:20]
            ce_scores = models.cross_encoder.predict([[query, x[0].text] for x in top_set])
            final = sorted([(top_set[i][0], (0.85 * ce_scores[i]) + (0.15 * top_set[i][1])) for i in range(len(top_set))], key=lambda x: x[1], reverse=True)
        else:
            final = [(x[0], x[1]) for x in scored]

        self.stats["latency"].append(time.perf_counter() - t0)

        # Extract IDs of retrieved items
        retrieved_ids = [f[0].id for f in final[:k]]

        # Update usage and true LRU last_access for ALL retrieved items
        for fid in retrieved_ids:
            if fid in self.L1.items:
                self.L1.items[fid].usage += 1
                self.L1.items[fid].last_access = time.time()
            elif fid in self.L2.items:
                self.L2.items[fid].usage += 1
                self.L2.items[fid].last_access = time.time()

        return retrieved_ids

# ============================================
# SECTION 5: DATASET GENERATION (Synthetic + Real)
# ============================================

def generate_dataset(n, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)  # optional strictness

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

def download_sample_bgl_log(target_file="BGL_sample.log"):
    if os.path.exists(target_file):
        print(f"File {target_file} already exists, using it.")
        return target_file

    # Correct URL for BGL 2k sample
    url = "https://raw.githubusercontent.com/logpai/loghub/master/BGL/BGL_2k.log"
    try:
        print(f"Downloading BGL sample from {url}...")
        urllib.request.urlretrieve(url, target_file)
        print(f"Downloaded {target_file}")
        return target_file
    except Exception as e:
        print(f"Download failed: {e}")

    # Fallback synthetic log (includes Node-* for compatibility)
    print("Creating synthetic fallback log...")
    with open(target_file, 'w') as f:
        f.write("- 2023-01-01 INFO Node-0 Kernel: System started\n")
        f.write("- 2023-01-01 INFO Node-1 Kernel: CPU online\n")
        f.write("FATAL 2023-01-01 ERROR Node-2 Kernel: panic: kernel panic\n")
        f.write("- 2023-01-01 INFO Node-3 Memory: 16GB available\n")
        f.write("ALERT 2023-01-01 ERROR Node-4 Network: link down\n")
        for i in range(10):
            f.write(f"- 2023-01-01 INFO Node-{i} Service: heartbeat\n")
    return target_file

def load_real_logs(filepath: str, max_entries: int = 5000):
    # Patterns for BGL logs (including Node-* for fallback compatibility)
    bgl_patterns = [
        r'R\d+-M\d+',           # node name in real BGL
        r'Node-\d+',            # node name in fallback
        r'FATAL|ERROR|panic',   # severity
        r'core\.\d+',           # core number
        r'0x[0-9a-fA-F]+',      # hex codes
        r'Kernel|Linux',        # keywords
    ]
    query_templates = [
        "Find the log entry containing {focus}",
        "Show me the system logs regarding {focus}",
        "What was the recorded error status involving {focus}?",
        "Retrieve the log entry with {focus}"
    ]

    facts = {}
    queries = {}
    with open(filepath, 'r') as f:
        for idx, line in enumerate(f, start=1):
            if idx > max_entries:
                break
            line = line.strip()
            if not line:
                continue
            lower = line.lower()
            if any(k in lower for k in ['fatal', 'error', 'panic', 'fail']):
                importance = 0.95
            else:
                importance = 0.5
            facts[idx] = {"text": line, "importance": importance}
            entities = router.extract_entities(line, bgl_patterns)
            if entities:
                # Use up to 2 entities to create a specific query for fair MRR evaluation
                ent_list = list(entities)
                focus_entities = random.sample(ent_list, min(2, len(ent_list)))
                focus = " and ".join(focus_entities)
                queries[idx] = random.choice(query_templates).format(focus=focus)
            else:
                queries[idx] = f"Retrieve log entry {idx}"
    return facts, queries

# ============================================
# SECTION 6: BENCHMARK SUITE
# ============================================

def run_publication_benchmark(num_seeds=5):
    scenarios = [
        {"name": "Scenario A: Overflow", "n": 3000, "l1": 500, "l2": 5000},
        {"name": "Scenario B: Saturation (Forgetting)", "n": 15000, "l1": 500, "l2": 5000},
        {"name": "Scenario C: Control (High-Cap)", "n": 5000, "l1": 1000, "l2": 14000}
    ]
    modes = ["full", "oracle_unbounded", "no_ce", "no_gate", "lru"]
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
                engine = HTMEngine(sc['l1'], sc['l2'], mode=mode, total_facts_estimate=sc['n'] + 5000)

                # Ingestion
                for fid, d in facts.items():
                    engine.ingest(fid, d['text'], d['importance'])

                # Evaluation (Active: recent 100, History: first 100)
                for label, ids in [("Active", range(sc['n']-100, sc['n']+1)), ("History", range(1, 101))]:
                    engine.stats["latency"] = []
                    mrr_vals = []
                    for sid in ids:
                        res = engine.retrieve(queries[sid])
                        mrr_vals.append(1 / (res.index(sid) + 1) if sid in res else 0)

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

def run_real_log_benchmark(log_file: str, max_entries=5000):
    random.seed(42)  # deterministic sampling
    print(f"\n[REAL LOG BENCHMARK: {log_file}]")
    facts, queries = load_real_logs(log_file, max_entries)
    n = len(facts)
    l1_cap = 500
    l2_cap = 5000
    modes = ["full", "oracle_unbounded", "lru"]
    results = []

    for mode in modes:
        print(f"  > Mode: {mode.upper()}")
        engine = HTMEngine(l1_cap, l2_cap, mode=mode, total_facts_estimate=n + 1000)
        for fid, d in facts.items():
            engine.ingest(fid, d['text'], d['importance'])

        # Evaluate on a random subset of 200 facts to keep CPU time reasonable
        eval_ids = random.sample(list(range(1, n+1)), min(n, 200))
        engine.stats["latency"] = []
        mrr_vals = []
        for sid in eval_ids:
            res = engine.retrieve(queries[sid])
            mrr_vals.append(1 / (res.index(sid) + 1) if sid in res else 0)

        lat = (np.mean(engine.stats["latency"]) * 1000) if engine.stats["latency"] else np.nan
        results.append({
            "Mode": mode,
            "MRR": np.mean(mrr_vals),
            "Latency_ms": lat,
            "Essential_Lost": engine.stats["essential_lost"],
            "Pruned_Total": engine.stats["pruned_total"]
        })

    return pd.DataFrame(results)

def main():
    # 1. Synthetic benchmark
    results_df = run_publication_benchmark(num_seeds=5)

    print("\n" + "="*95)
    print("HTM-EAR: MULTI-SEED ROBUSTNESS ANALYSIS (N=15,000 | FINAL PRODUCTION SUITE)")
    print("="*95)

    scen_b_mask = results_df["Scenario"].str.contains("Scenario B")

    # TABLE 1: Retrieval Performance
    table_b_mrr = results_df[scen_b_mask].groupby(["Mode", "Temporal"])["MRR"].agg(['mean', 'std']).unstack()
    print("\n[TABLE 1: RETRIEVAL PERFORMANCE (MEAN +/- STD)]")
    print(table_b_mrr.to_string())

    # TABLE 2: System Robustness Metrics
    sys_stats = (results_df[scen_b_mask & (results_df["Temporal"] == "Active")]
                .groupby("Mode")[["Latency_ms", "Essential_Lost", "Pruned_Total"]]
                .agg(['mean', 'std']))
    print("\n[TABLE 2: SYSTEM ROBUSTNESS METRICS (MEAN +/- STD)]")
    print(sys_stats.to_string())

    # 2. Real log benchmark
    print("\n" + "="*95)
    print("REAL-WORLD LOG VALIDATION (BGL Dataset)")
    print("="*95)

    log_file = download_sample_bgl_log("BGL_sample.log")
    if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
        real_results = run_real_log_benchmark(log_file, max_entries=2000)
        print("\n[REAL LOG BENCHMARK RESULTS]")
        print(real_results.to_string())


        plt.figure(figsize=(10, 6))
        sns.barplot(data=real_results, x="Mode", y="MRR", hue="Mode", palette="Set2", legend=False)
        plt.title("HTM-EAR Performance on Real BGL Logs", fontsize=14, fontweight='bold')
        plt.ylabel("Mean Reciprocal Rank (MRR)")
        plt.tight_layout()
        plt.savefig("real_log_performance.png")
        plt.show()
    else:
        print("\n[WARNING] Could not obtain real log file. Run with synthetic data only.")

    # 3. Original plots (Scenario B)
    plt.figure(figsize=(14, 7))
    sns.set_style("whitegrid")
    sns.barplot(data=results_df[scen_b_mask], x="Mode", y="MRR", hue="Temporal",
                palette="coolwarm", errorbar="sd", capsize=.1)
    plt.title("Ablation Study: Precision Decay in Saturated Systems (Scenario B)", fontsize=14, fontweight='bold')
    plt.ylabel("Mean Reciprocal Rank (MRR)", fontsize=13)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig("precision_decay_v9.0.png")

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
    plt.title("Pareto Frontier: Active Retrieval vs. Phase Latency", fontsize=16, fontweight='bold')
    plt.axhspan(0, 0.6, color='red', alpha=0.1, label="Failure Zone")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pareto_frontier_v9.0.png")

    plt.show()

if __name__ == "__main__":
    main()