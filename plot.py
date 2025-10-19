import json
import matplotlib.pyplot as plt

with open("benchmark_results.json") as f:
    data = json.load(f)

plt.figure(figsize=(8, 5))
plt.plot(data["seq_len"], data["time_without_cache"], marker='o', label="Without KV Cache")
plt.plot(data["seq_len"], data["time_with_cache"], marker='s', label="With KV Cache")
plt.xlabel("Sequence Length")
plt.ylabel("Latency (seconds)")
plt.title("Latency vs Sequence Length (KV Caching)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# plot the results
