import math
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------------
# model and device specifications
L = 32                 # Number of layers
query_heads = 32       # Number of Query heads (not directly used in calculation)
kv_heads = 8           # Number of KV heads (not directly used as splitting cancels out))
h1 = 4096              # Embedding dimension (h1)
h2 = 14336             # Inner dimension (h2)
V = 128256             # Vocab size
GPU_mem = 80           # G PU memory(in GB)

# input specification
s = 1024               # Context length (s)

# quantization specifications (using FP16)
weight_bytes = 2       # num of bytes per param
kvcache_bytes = 2     # num of bytes per KV cache element


# -------------------------------------------------------------------------------------------
# Question 1a- Calculaton of model parameters, parameter memory, KV cache per request and Max batch size possible


# calculation for total model parameters-
attn_params = 4 * (h1 ** 2)   # attention block per layer: 4 * h1^2 (for Q, K, V and O projection)
ffn_params = 2 * h1 * h2     # feedforward block per layer: 2 * h1 * h2
param_per_layer = attn_params + ffn_params  # total params per layer
layers_params = L * param_per_layer    # for all layers

embed_params = V * h1     # embedding params

N_params = layers_params + embed_params      #total model params

# calculation for total parameter memory
param_mem_bytes = N_params * weight_bytes            #in bytes
param_mem_GB = param_mem_bytes / (1024**3)      #in GB

# calculation for KV Cache per request
kvcache_bytes_total = 2 * kvcache_bytes * L * h1 * s      # in bytes
kvcache_GB = kvcache_bytes_total / (1024**3)              #in GB

# calculation for max batch size possible
avlbl_mem = GPU_mem - param_mem_GB
max_batch_size = math.floor(avlbl_mem / kvcache_GB)      #take floor as batch size should be integer

# results of calculation for Q1a
print("----- Question 1(a) Results -----")
print("Total model parameters: {:.2f} Billion".format(N_params / 1e9))
print("Memory in bytes: ",param_mem_bytes)
print("Total parameter memory: {:.2f} GB".format(param_mem_GB))
print("KV cache per request: {:.2f} GB".format(kvcache_GB))
print("Max batch size possible: {}".format(max_batch_size))

# ----------------------
# Question 1b- Tensor Parallelism
# when using TP with dimension X, both model memory and KV cache are divided equally among X GPU
# max batch size per GPU: B_max = (80 - (total param memory in GB)/X) / ((kv cache in GB)/X)
def max_batchsize_TP(X):
    return math.floor((80 * X - param_mem_GB) / kvcache_GB)

# example-   X=2 GPUs
X_eg = 2
print("\n----- Question 1(b) Results -----")
print("For Tensor Parallelism (TP) dimension X = {}:".format(X_eg))
print("Max batch size possible per GPU: {}".format(max_batchsize_TP(X_eg)))

# ----------------------
# Question 1d- Calculatin for arithmetic intensity and Plots
# from lecture slides, each multiply-add is counted as 2 FLOPs
# For FP16, BPE = weight_bytes = 2.

# prefill_flops = 4N^2d
# prefill_memory = (4N^2 + 4Nd)BPE
def calc_ai_prefill(N, d, BPE):
    return (4 * (N**2) * d) / ((4 * N**2 + 4 * N * d) * BPE)     # AI_prefill = prefill_flops / prefill_memory


# decode_flops = 4Nd
# decode_memory = (2d + 2Nd + 4N)BPE
def calc_ai_decode(N, d, BPE):
    return (4 * N * d) / ((2 * d + 2 * N * d + 4 * N) * BPE)       # AI_decode = decode_flops / decode_memory

# param values
N_val = 1024
d_val = 128
BPE = weight_bytes

ai_prefill = calc_ai_prefill(N_val, d_val, BPE)
ai_decode = calc_ai_decode(N_val, d_val, BPE)

print("\n----- Question 1(d) Arithmetic Intensity -----")
print("Arithmetic Intensity (Prefill) for N=1024, d=128: {:.2f} FLOPs/byte".format(ai_prefill))
print("Arithmetic Intensity (Decode) for N=1024, d=128: {:.2f} FLOPs/byte".format(ai_decode))

# plot of AI_prefill vs. context length for prefill with batch size 1
N_value_range = np.linspace(256, 8192, 50)
ai_prefill_values = [calc_ai_prefill(n, d_val, BPE) for n in N_value_range]

plt.figure(figsize=(8, 5))
plt.plot(N_value_range, ai_prefill_values, marker='o')
plt.xlabel('Context Length (N)')
plt.ylabel('Arithmetic Intensity (FLOPs/byte) for Prefill')
plt.title('Arithmetic Intensity vs. Context Length (Prefill, d=128, FP16)')
plt.grid(True)
plt.show()

# plot of AI_decode vs. batch size for decode attention
batch_size_range = np.arange(1, 101)
ai_decode_values = [calc_ai_decode(N_val, d_val, BPE) for _ in batch_size_range]

plt.figure(figsize=(8, 5))
plt.plot(batch_size_range, ai_decode_values, marker='x', linestyle='--')
plt.xlabel('Batch Size')
plt.ylabel('Arithmetic Intensity (FLOPs/byte) for Decode')
plt.title('Arithmetic Intensity vs. Batch Size (Decode, N=1024, d=128, FP16)')
plt.grid(True)
plt.show()
# As per request arithmetic intensity in decode mode does not depend on batch size,the AI remains constant. We show this by plotting a constant line.



