# ⚡ Transformer Inference Optimization using KV-Cache

This project focuses on optimizing **Transformer-based model inference** by implementing and profiling **Key-Value (KV) Caching** to reduce redundant computation during autoregressive text generation.

---

## 🚀 Overview
Transformers recompute attention keys and values for all past tokens at each decoding step, leading to significant latency in long-sequence inference.  
This project implements **KV-Cache optimization** to store and reuse past attention states, enabling **faster token generation** without affecting output quality.

---

## 🧠 Key Contributions
- Implemented **KV-Cache mechanism** for Transformer decoder layers.  
- Profiled decoding performance with and without caching for variable sequence lengths.  
- Measured improvements in **throughput and latency**, showing substantial speed-up in long-context scenarios.  
- Analyzed trade-offs between memory footprint and inference efficiency.

---

## 📈 Results Summary
| Sequence Length | Inference Type | Avg. Decoding Time (s) | Speed-up |
|-----------------|----------------|-------------------------|-----------|
| 64 tokens | Without KV-Cache | 1.00× | — |
| 256 tokens | With KV-Cache | **≈3.2× faster** | ✔️ |

> KV-Cache significantly reduces recomputation overhead, especially for long sequences in autoregressive decoding.

---

## 🧩 Technical Details
- **Model:** Decoder-only Transformer (GPT-like architecture).  
- **Optimization:** KV-cache at each attention layer (keys & values persisted across steps).  
- **Framework:** PyTorch with CUDA profiling for latency measurement.  

---

## 🛠️ Requirements
```bash
pip install torch numpy matplotlib tqdm
