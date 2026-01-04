# HTM-EAR  
**Hierarchical Tiered Memory with Essential-Aware Retention**

---

## Project Overview

This repository contains the implementation of **HTM-EAR**, a memory management system designed to study how information is stored, forgotten, and preserved under **explicit memory capacity constraints**. The system focuses on **controlled forgetting under memory saturation** and explicitly tracks whether **essential information is ever lost**.

HTM-EAR is evaluated as an **infrastructure-level memory substrate**.  
It does **not** target downstream task accuracy or benchmark performance.

---

## Project

### HTM-EAR System (`HTM-EAR.py`)

A complete experimental framework that simulates **hierarchical memory** with **policy-driven retention and eviction**, enabling systematic analysis of memory behavior under saturation.

---

## Key Features

- Hierarchical memory structure with **working memory (L1)** and **history memory (L2)**
- Capacity-based overflow handling and pruning
- Policy-controlled forgetting mechanism
- Explicit tracking of **essential vs non-essential** information
- Semantic routing gate for memory access decisions
- Multi-seed experimental evaluation for robustness

---

## Experiments

The system evaluates three controlled scenarios:

### 1. Overflow Scenario  
Memory exceeds **working memory (L1)** capacity.

### 2. Forgetting Scenario  
Memory exceeds **total memory (L1 + L2)** capacity, triggering controlled forgetting.

### 3. Control Scenario  
Memory remains within capacity limits.

Each scenario is evaluated across **multiple random seeds** to ensure stability and reproducibility.

---

## Metrics

The following metrics are reported:

- **Active memory retrieval performance** (recent information)
- **Historical memory retrieval performance** (older information)
- **Essential memory loss** (count of essential items permanently lost)
- **Pruned entries** (total number of evicted items)
- **Latency statistics** (active retrieval path only)

> *Retention is evaluated implicitly through essential memory survival and historical retrieval behavior.*

Final aggregated results and tables are provided in **`HTM-EAR-OUTPUT.docx`**.

---

## Usage

All experiments can be reproduced by running:

```bash
pip install -r requirements.txt
python HTM-EAR.py
Technical Details
Framework: Python

Retrieval: Dense vector embeddings with FAISS (inner product search)

Execution: CPU-based by default

GPU: If available, affects runtime only (not experimental outcomes)

Files
HTM-EAR.py — Core implementation and experimental framework

HTM-EAR-OUTPUT.docx — Final experimental results and tables

requirements.txt — Python dependencies

Author
Shubham Kumar Singh

License
MIT License

Copyright (c) 2026 Shubham Kumar Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

