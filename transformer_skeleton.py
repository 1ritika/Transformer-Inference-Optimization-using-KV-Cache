import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import argparse
import os
from collections import OrderedDict


class SingleHeadAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # TODO: Implement single-head attention
        # Create query, key, value, and output projection layers
        # Remember to set bias=False for query, key, value projections. 
        # Example : self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # query projection layer
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # key projection layer
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # value projection layer
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # output projection layer

    def forward(self, x, past_kv=None, use_cache=False):
         # TODO: Implement the forward pass for single-head attention
        # 1. Compute query, key, value projections
        # 2. Handle KV caching if use_cache=True
        # 3. Compute attention scores with scaling #hint use torch.bmm
        # 4. Apply causal masking #hint: use torch.triu
        # 5. Apply softmax to get attention weights
        # 6. Compute context vector #hint: use torch.bmm
        # 7. Compute output and return
        
        # Your code here
        
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # concatenate cached keys/values if available
        if use_cache and past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=1)
            v = torch.cat([past_kv[1], v], dim=1)

        d_k = self.d_model ** 0.5  # scaling factor
        scores = torch.bmm(q, k.transpose(1, 2)) / d_k

        # apply causal masking
        mask = torch.triu(torch.ones(scores.shape[-2:], device=scores.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, v)

        output = self.W_o(context)


        # Placeholder return - replace with your implementation
        return output, (k, v) if use_cache else None


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # TODO: Implement feed-forward network
        # Create two linear layers 
        
        # Your code here
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        # TODO: Implement the forward pass for feed-forward network
        # Two linear layers with ReLU activation in between
        # Your code here
        # first linear transformation
        x = self.fc1(x)

        # relu activation
        x = F.relu(x)

        # second linear transformation
        x = self.fc2(x)


        # Placeholder return - replace with your implementation
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # TODO: Implement decoder layer
        # Create attention, feed-forward, layer norms, and dropout
        
        # Your code here
        self.attention = SingleHeadAttention(d_model)  #  attention layer
        self.feed_forward = FeedForward(d_model, d_ff)  #  feed-forward network
        self.norm1 = nn.LayerNorm(d_model)            # layer norm 1
        self.norm2 = nn.LayerNorm(d_model)            # layer norm 2
        self.dropout = nn.Dropout(dropout)            # dropout layer

        
    def forward(self, x, past_kv=None, use_cache=False):
        # TODO: Implement the forward pass for decoder layer
        # 1. Apply layer norm before attention
        # 2. Apply attention followed by dropout
        # 3. Apply layer norm before feed-forward
        # 4. Apply feed-forward then residual connection
        
        # Your code here
        x_norm = self.norm1(x)  # normalize input
        attn_output, new_kv = self.attention(x_norm, past_kv, use_cache)
        x = x + self.dropout(attn_output)  # add residual connection

        x_norm = self.norm2(x)  # normalize before feed-forward
        ff_output = self.feed_forward(x_norm)
        x = x + self.dropout(ff_output)  # add residual connection


        # Placeholder return - replace with your implementation
        return x, new_kv

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, num_layers, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # TODO: Implement decoder-only transformer
        # Create token embedding, positional embedding, decoder layers, output projection
        
        # Your code here

        self.token_embedding = nn.Embedding(vocab_size, d_model)  # token embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)   # positional embeddings

        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ff) for _ in range(num_layers)])  # stack of decoder layers
        self.norm = nn.LayerNorm(d_model)         # final layer normalization
        self.output_projection = nn.Linear(d_model, vocab_size)  # project to vocabulary size

    
    def forward(self, input_ids, past_kv=None, use_cache=False):
        # TODO: Implement the forward pass for decoder-only transformer
        # 1. Get token embeddings
        # 2. Add positional embeddings (handle position offsets for cached generation)
        # 3. Pass through decoder layers
        # 4. Apply final layer norm
        # 5. Project to vocabulary
        
        # Your code here
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)  # generate positions

        # sum token and positional embeddings
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)

        new_past_kv = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            x, new_kv = layer(x, past_kv[i] if past_kv else None, use_cache)
            if use_cache:
                new_past_kv.append(new_kv)

        x = self.norm(x)  # final normalization
        logits = self.output_projection(x)  # project to vocab space


        return logits, new_past_kv
    
    def generate(self, input_ids, max_new_tokens, temperature=1.0, use_cache=True):
        # TODO: Implement the generation method
        # 1. Start with the input sequence
        # 2. Iteratively generate new tokens
        # 3. Use temperature for sampling
        # 4. Use KV caching for efficiency
        
        # Your code here
        device = input_ids.device  # ensure correct device
        past_kv = None  # initialize kv cache

        generated_ids = input_ids.clone()  # start with initial input

        for _ in range(max_new_tokens):
            # use only last token if caching is enabled
            input_chunk = generated_ids[:, -1:] if use_cache else generated_ids
            logits, past_kv = self.forward(input_chunk, past_kv=past_kv, use_cache=use_cache)
            
            # get logits for the last token
            logits = logits[:, -1, :]
            
            # apply temperature scaling
            logits = logits / temperature
            
            # sample the next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # append the new token to the sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

        # Placeholder return - replace with your implementation
        return generated_ids


# ================= DO NOT MODIFY CODE BELOW THIS LINE =================
# Evaluation harness code - do not modify

def load_test_cases(filepath):
    """Load test cases from a file."""
    with open(filepath, 'r') as f:
        test_cases = json.load(f)
    
    # Convert lists back to tensors
    for case in test_cases:
        case['input_ids'] = torch.tensor(case['input_ids'])
        case['expected_logits_no_cache'] = torch.tensor(case['expected_logits_no_cache'])
        case['expected_logits_with_cache'] = torch.tensor(case['expected_logits_with_cache'])
        case['expected_logits_sequential'] = torch.tensor(case['expected_logits_sequential'])
    
    return test_cases

def evaluate_model(model, test_cases, atol=1e-2, with_kv=False):
    """Evaluate model against test cases."""
    model.eval()
    results = []
    
    for i, case in enumerate(test_cases):
        input_ids = case['input_ids']
        expected_logits_no_cache = case['expected_logits_no_cache']
        expected_logits_with_cache = case['expected_logits_with_cache']
        expected_logits_sequential = case['expected_logits_sequential']
        
        with torch.no_grad():
            # Test without caching
            logits_no_cache, _ = model(input_ids, use_cache=False)
            no_cache_match = torch.allclose(logits_no_cache, expected_logits_no_cache, atol=atol)
            
            if with_kv:
                # Test with caching (full sequence)
                logits_with_cache, _ = model(input_ids, use_cache=True)
                with_cache_match = torch.allclose(logits_with_cache, expected_logits_with_cache, atol=atol)

                cache_nocache_match = torch.allclose(logits_no_cache, logits_with_cache, atol=atol)
            
        
        result = {
            'test_case': i + 1,
            'no_cache_match': no_cache_match,
            'with_cache_match': with_cache_match if with_kv else None,
            'cache_nocache_match': cache_nocache_match if with_kv else None,
            'all_match': no_cache_match and (with_cache_match and cache_nocache_match if with_kv else no_cache_match)
        }
        
        if not result['all_match']:
            # Calculate error metrics for debugging
            if not no_cache_match:
                result['no_cache_max_error'] = torch.max(torch.abs(logits_no_cache - expected_logits_no_cache)).item()
            if with_kv and not with_cache_match:
                result['with_cache_max_error'] = torch.max(torch.abs(logits_with_cache - expected_logits_with_cache)).item()
            if with_kv and not cache_nocache_match:
                result['cache_nocache_max_error'] = torch.max(torch.abs(logits_no_cache - logits_with_cache)).item()
        
        results.append(result)
    
    # Overall results
    all_passed = all(r['all_match'] for r in results)
    pass_rate = sum(r['all_match'] for r in results) / len(results)
    
    summary = {
        'all_passed': all_passed,
        'pass_rate': pass_rate,
        'num_test_cases': len(test_cases),
        'num_passed': sum(r['all_match'] for r in results),
        'detailed_results': results
    }
    
    return summary

def benchmark_performance(model, input_ids, num_new_tokens=20, use_cache=True, num_runs=3):
    """Benchmark model performance."""
    model.eval()
    
    # Warm-up run
    model.generate(input_ids, num_new_tokens, use_cache=use_cache)
    
    # Timed runs
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        model.generate(input_ids, num_new_tokens, use_cache=use_cache)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    return avg_time


def main():
    parser = argparse.ArgumentParser(description='Transformer Evaluation Harness')
    parser.add_argument('--mode', type=str, default='run', choices=['generate', 'evaluate', 'kv_evaluate', 'benchmark', 'run'], 
                        help='Mode to run in')
    parser.add_argument('--weights', type=str, default='reference_weights.pt', 
                        help='Path to weights file')
    parser.add_argument('--model_state_dict', type=str, default='model_state_dict.pt', 
                        help='Path to model state dictionary file')
    parser.add_argument('--test_cases', type=str, default='test_cases.json', 
                        help='Path to test cases file')
    parser.add_argument('--vocab_size', type=int, default=1000, 
                        help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=50, 
                        help='Model dimension')
    parser.add_argument('--d_ff', type=int, default=100, 
                        help='Feed-forward dimension')
    parser.add_argument('--num_layers', type=int, default=2, 
                        help='Number of decoder layers')
    parser.add_argument('--max_seq_len', type=int, default=128, 
                        help='Maximum sequence length')
    
    args = parser.parse_args()
    
    if args.mode == 'generate':
        # Generate evaluation harness -- not accessible to students
        generate_evaluation_harness(args.vocab_size, args.d_model, args.d_ff, args.num_layers, args.max_seq_len)
    
    elif args.mode == 'evaluate':
        # Evaluate a model
        if not os.path.exists(args.model_state_dict):
            print(f"Error: Model state dictionary file {args.model_state_dict} not found.")
            return

        if not os.path.exists(args.test_cases):
            print(f"Error: Test cases file {args.test_cases} not found.")
            return
        
        # Create model
        model = DecoderOnlyTransformer(args.vocab_size, args.d_model, args.d_ff, args.num_layers, args.max_seq_len)
        
        try:
            model.load_state_dict(torch.load(args.model_state_dict))
            print(f"Successfully loaded model state dictionary from {args.model_state_dict}")
        except Exception as e:
            print(f"Error loading model state dictionary: {e}")
            return
        
        # Load test cases
        test_cases = load_test_cases(args.test_cases)
        print(f"Test cases loaded from {args.test_cases}")
        
        # Evaluate model
        results = evaluate_model(model, test_cases, with_kv=False)
        
        # Print results
        print(f"Evaluation Results:")
        print(f"  Num test cases: {results['num_test_cases']}")
        print(f"  All tests passed: {results['all_passed']}")
        print(f"  Pass rate: {results['pass_rate'] * 100:.2f}% ({results['num_passed']}/{results['num_test_cases']})")
        # Print result stats - each test case with pass/fail info
        #print(f"  Detailed results: {results['detailed_results']}")

        
        if not results['all_passed']:
            print("\nFailed test cases:")
            for i, result in enumerate(results['detailed_results']):
                if not result['all_match']:
                    print(f"  Test case {result['test_case']}:")
                    if not result.get('no_cache_match', True):
                        print(f"    No cache: Failed (max error: {result.get('no_cache_max_error', 'N/A')})")

    elif args.mode == 'kv_evaluate':
        # Evaluate a model with kv cache against no_kv_cache
        if not os.path.exists(args.model_state_dict):
            print(f"Error: Model state dictionary file {args.model_state_dict} not found.")
            return

        if not os.path.exists(args.test_cases):
            print(f"Error: Test cases file {args.test_cases} not found.")
            return
        
        # Create model
        model = DecoderOnlyTransformer(args.vocab_size, args.d_model, args.d_ff, args.num_layers, args.max_seq_len)
        
        try:
            model.load_state_dict(torch.load(args.model_state_dict))
            print(f"Successfully loaded model state dictionary from {args.model_state_dict}")
        except Exception as e:
            print(f"Error loading model state dictionary: {e}")
            return
        
        # Load test cases
        test_cases = load_test_cases(args.test_cases)
        print(f"Test cases loaded from {args.test_cases}")
        
        # Evaluate model
        results = evaluate_model(model, test_cases, with_kv=True)
        
        # Print results
        print(f"Evaluation Results:")
        print(f"  Num test cases: {results['num_test_cases']}")
        print(f"  All tests passed: {results['all_passed']}")
        print(f"  Pass rate: {results['pass_rate'] * 100:.2f}% ({results['num_passed']}/{results['num_test_cases']})")
        #detailed results
        #print(f"  Detailed results: {results['detailed_results']}")

        if not results['all_passed']:
            print("\nFailed test cases:")
            for i, result in enumerate(results['detailed_results']):
                if not result['all_match']:
                    print(f"  Test case {result['test_case']}:")
                    if not result.get('no_cache_match', True):
                        print(f"    No cache: Failed (max error: {result.get('no_cache_max_error', 'N/A')})")
                    if not result.get('with_cache_match', True):
                        print(f"    With cache: Failed (max error: {result.get('with_cache_max_error', 'N/A')})")
    
    elif args.mode == 'benchmark':
        # Benchmark model performance
        if not os.path.exists(args.model_state_dict):
            print(f"Error: Model state dictionary file {args.model_state_dict} not found.")
            return

        # Create model
        model = DecoderOnlyTransformer(args.vocab_size, args.d_model, args.d_ff, args.num_layers, args.max_seq_len)
        

        # Load model state dict
        try:
            model.load_state_dict(torch.load(args.model_state_dict))
            print(f"Successfully loaded model state dictionary from {args.model_state_dict}")
        except Exception as e:
            print(f"Error loading model state dictionary: {e}")
            return

        
        # Create sample input
        input_ids = torch.randint(0, args.vocab_size, (1, 10))
        
        # Benchmark with and without caching
        print("Benchmarking...")
        time_without_cache = benchmark_performance(model, input_ids, use_cache=False)
        time_with_cache = benchmark_performance(model, input_ids, use_cache=True)
        
        print(f"Results for default:")
        print(f"  Without KV cache: {time_without_cache:.4f} seconds")
        print(f"  With KV cache: {time_with_cache:.4f} seconds")
        print(f"  Speedup: {time_without_cache / time_with_cache:.2f}x")

        # Begin added code for plotting
        #seq_len = [10, 50, 100, 500, 1000]                # sequence len to test
        #time_with_cache = []
        #time_without_cache = []
        #tested_seq =[]

        #for seql in seq_len:
        #    if seql > args.max_seq_len:
        #        print(f"Skipping seq_len {seql} as it exceeds max_seq_len ({args.max_seq_len})")
        #        continue

        #    input_ids = torch.randint(0, args.vocab_size, (1, seql))

        #    time_no_cache = benchmark_performance(model, input_ids, use_cache=False)
        #    time_cache = benchmark_performance(model, input_ids, use_cache=True)

        #    time_without_cache.append(time_no_cache)
        #    time_with_cache.append(time_cache)
        #    tested_seq.append(seql)

        #    print(f"Sequence Length: {seql}")
        #    print(f"  Without KV cache: {time_no_cache:.4f} seconds")
        #    print(f"  With KV cache: {time_cache:.4f} seconds")
        #    print(f"  Speedup: {time_no_cache / time_cache:.2f}x")


        # save data for plotting
        #with open("benchmark_results.json", "w") as f:
        #    json.dump({
        #        "seq_len": tested_seq,
        #        "time_without_cache": time_without_cache,
        #        "time_with_cache": time_with_cache
        #    }, f)

        #End added code
    
    elif args.mode == 'run':
        # Just a debugging mode

        # Default mode: generate harness if files don't exist, then evaluate and benchmark
        if not os.path.exists('model_state_dict.pt') or not os.path.exists('test_cases.json'):
            generate_evaluation_harness(args.vocab_size, args.d_model, args.d_ff, args.num_layers, args.max_seq_len)
        
        # Create model
        model = DecoderOnlyTransformer(args.vocab_size, args.d_model, args.d_ff, args.num_layers, args.max_seq_len)
        
        # Load model state dict
        state_dict = torch.load('model_state_dict.pt')

        # Print specific weights from state dict
        #print("From state dict - layer 0 Wq weight:")
        #print(state_dict['layers.0.attention.W_q.weight'])
        #print("From state dict - layer 1 Wk weight:")
        #print(state_dict['layers.1.attention.W_k.weight'])

        try:
            model.load_state_dict(state_dict)
            print(f"Successfully loaded model state dictionary from model_state_dict.pt")
        except Exception as e:
            print(f"Error loading model state dictionary: {e}")
            return
 
        print("Weights loaded from {}".format(args.model_state_dict))

        # print the structure of state_dict
        #print("State dict structure:")
        #print(state_dict.keys())

        # Verify they're the same
        print("Weights match for layer 0 Wq:", 
        torch.allclose(state_dict['layers.0.attention.W_q.weight'], model.layers[0].attention.W_q.weight))
        print("Weights match for layer 1 Wk:", 
        torch.allclose(state_dict['layers.1.attention.W_k.weight'], model.layers[1].attention.W_k.weight))
        
        # Load test cases
        test_cases = load_test_cases('test_cases.json')
        print(f"Test cases loaded from test_cases.json")

        # Print the first test case
        print("First test case:")
        print(test_cases[0].keys())
        # print the tensor shape for each key
        for key in test_cases[0].keys():
            print(key, test_cases[0][key].shape)

        # Evaluate model
        print("\nEvaluating model...")
        results = evaluate_model(model, test_cases)
        
        # Print evaluation results
        print(f"Evaluation Results:")
        print(f"  All tests passed: {results['all_passed']}")
        print(f"  Pass rate: {results['pass_rate'] * 100:.2f}% ({results['num_passed']}/{results['num_test_cases']})")
        
        # Benchmark
        print("\nBenchmarking performance...")
        input_ids = torch.randint(0, args.vocab_size, (1, 100))
        
        time_without_cache = benchmark_performance(model, input_ids, use_cache=False)
        time_with_cache = benchmark_performance(model, input_ids, use_cache=True)
        
        print(f"Performance Results:")
        print(f"  Without KV cache: {time_without_cache:.4f} seconds")
        print(f"  With KV cache: {time_with_cache:.4f} seconds")
        print(f"  Speedup: {time_with_cache > 0 and time_without_cache / time_with_cache:.2f}x")

if __name__ == "__main__":
    main()