#!/usr/bin/env python3
"""
Model profiler for comparing optimized implementations against baseline models.

This profiler compares your optimized implementations (e.g., xlstm.py) against 
the baseline models in the model/ directory.

Usage Examples:
    # Compare optimized xLSTM vs baseline
    python model_profiler.py --model xlstm
    
    # Compare with custom settings
    python model_profiler.py --model xlstm --batch-size 16 --hidden-size 512
    
    # Quick test
    python model_profiler.py --model xlstm --num-runs 10
    
    # When you create mamba2.py optimized version:
    python model_profiler.py --model mamba2
"""

import torch
import time
import numpy as np
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable

sys.path.insert(0, str(Path(__file__).parent))

from model.inference_model import create_layer, LayerType

class ModelProfiler:
    """Simple profiler for any PyTorch model."""
    
    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._setup_device()
    
    def _setup_device(self):
        """Setup device optimizations."""
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"Device: {self.device} ({torch.cuda.get_device_name()})")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print(f"Device: {self.device}")
    
    def benchmark_model(
        self, 
        model: torch.nn.Module,
        input_fn: Callable,
        num_runs: int = 50,
        warmup_runs: int = 10,
        name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Benchmark a model with given input function.
        
        Args:
            model: PyTorch model to benchmark
            input_fn: Function that returns (inputs, states) tuple for the model
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            name: Name for display
        """
        print(f"\nBenchmarking {name}...")
        
        model.eval()
        
        # Memory before
        mem_before = self._get_memory_usage()
        
        # Warmup
        for _ in range(warmup_runs):
            inputs, states = input_fn()
            with torch.no_grad():
                output = self._run_model(model, inputs, states)
            self._sync()
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            inputs, states = input_fn()
            
            self._sync()
            start = time.perf_counter()
            
            with torch.no_grad():
                output = self._run_model(model, inputs, states)
            
            self._sync()
            end = time.perf_counter()
            
            times.append(end - start)
        
        # Memory after
        mem_after = self._get_memory_usage()
        
        # Statistics
        times = np.array(times)
        results = {
            'name': name,
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'median_time': np.median(times),
            'memory_delta': mem_after - mem_before,
            'throughput_per_sec': 1.0 / np.mean(times),
            'times': times
        }
        
        self._print_results(results)
        return results
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated() / 1024**2
        return 0
    
    def _sync(self):
        """Synchronize device."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
    
    def _run_model(self, model, inputs, states):
        """Run model with proper state handling."""
        # If states is None, try to initialize it
        if states is None and hasattr(model, 'init_state'):
            batch_size = inputs.shape[0] if hasattr(inputs, 'shape') else 1
            states = model.init_state(batch_size, self.device)
        
        # Try different calling conventions
        try:
            if states is not None:
                return model(inputs, states)
            else:
                return model(inputs)
        except TypeError:
            # Try returning both output and states
            try:
                output, new_states = model(inputs, states)
                return output
            except:
                # Last resort - just pass inputs
                return model(inputs)
    
    def _print_results(self, results: Dict[str, Any]):
        """Print benchmark results."""
        print(f"  Mean time:    {results['mean_time']*1000:.3f} ± {results['std_time']*1000:.3f} ms")
        print(f"  Median time:  {results['median_time']*1000:.3f} ms")
        print(f"  Min time:     {results['min_time']*1000:.3f} ms")
        print(f"  Throughput:   {results['throughput_per_sec']:.1f} inferences/sec")
        if results['memory_delta'] != 0:
            print(f"  Memory delta: {results['memory_delta']:.2f} MB")
    
    def compare_models(self, model_configs: list, input_fn: Callable, **kwargs) -> Dict[str, Any]:
        """
        Compare multiple models.
        
        Args:
            model_configs: List of (model, name) tuples
            input_fn: Function that returns (inputs, states) for models
            **kwargs: Additional arguments for benchmark_model
        """
        print("="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        results = {}
        for model, name in model_configs:
            results[name] = self.benchmark_model(model, input_fn, name=name, **kwargs)
        
        # Summary comparison
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        baseline_time = None
        for name, result in results.items():
            if baseline_time is None:
                baseline_time = result['mean_time']
                print(f"{name:20s}: {result['mean_time']*1000:6.3f} ms (baseline)")
            else:
                speedup = baseline_time / result['mean_time']
                print(f"{name:20s}: {result['mean_time']*1000:6.3f} ms ({speedup:4.2f}x)")
        
        return results

# Model factories for baseline vs optimized comparison
def create_baseline_model(model_name: str, hidden_size: int = 256, **kwargs):
    """Create baseline model from model/ directory."""
    layer_type_map = {
        'xlstm': LayerType.XLSTM,
        'mamba2': LayerType.MAMBA2,
        'retnet': LayerType.RETNET,
        'hawk': LayerType.HAWK
    }
    
    if model_name not in layer_type_map:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(layer_type_map.keys())}")
    
    return create_layer(layer_type_map[model_name], hidden_size, **kwargs)

def create_optimized_model(model_name: str, hidden_size: int = 256, **kwargs):
    """Create optimized model from root directory files."""
    if model_name == 'xlstm':
        from xlstm import create_optimized_xlstm
        return create_optimized_xlstm(
            vocab_size=1000,
            hidden_size=hidden_size,
            num_layers=1,
            compile_model=False,  # Fair comparison without compilation
            use_mixed_precision=False,  # Disable for debugging
            use_gradient_checkpointing=False,  # Disable for debugging
            **kwargs
        )
    elif model_name == 'mamba2':
        # Import your optimized mamba2 when you create mamba2.py
        try:
            from mamba2 import create_optimized_mamba2
            return create_optimized_mamba2(hidden_size=hidden_size, **kwargs)
        except ImportError:
            raise ImportError(f"Optimized {model_name}.py not found. Create it first!")
    elif model_name == 'retnet':
        try:
            from retnet import create_optimized_retnet  
            return create_optimized_retnet(hidden_size=hidden_size, **kwargs)
        except ImportError:
            raise ImportError(f"Optimized {model_name}.py not found. Create it first!")
    elif model_name == 'hawk':
        try:
            from hawk import create_optimized_hawk
            return create_optimized_hawk(hidden_size=hidden_size, **kwargs)
        except ImportError:
            raise ImportError(f"Optimized {model_name}.py not found. Create it first!")
    else:
        raise ValueError(f"Unknown model: {model_name}")

# Input generators
def single_timestep_input(batch_size: int = 16, hidden_size: int = 256, device=None):
    """Generate single timestep input."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def input_fn():
        x = torch.randn(batch_size, hidden_size, device=device)
        # Most models need states, create dummy ones
        states = None  # Will be handled by the model's init_state if needed
        return x, states
    
    return input_fn

def token_input(batch_size: int = 16, seq_len: int = 32, vocab_size: int = 1000, device=None):
    """Generate token input for language models."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def input_fn():
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        states = None
        return x, states
    
    return input_fn

def main():
    """CLI interface for the profiler."""
    parser = argparse.ArgumentParser(
        description="Compare optimized model implementations against baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python model_profiler.py --model xlstm                    # Compare xlstm.py vs model/xlstm.py
  python model_profiler.py --model xlstm --batch-size 32   # Custom batch size
  python model_profiler.py --model mamba2                  # When you create mamba2.py
        """
    )
    parser.add_argument("--model", 
                       choices=["xlstm", "mamba2", "retnet", "hawk"],
                       required=True,
                       help="Model to compare (optimized vs baseline)")
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden size")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length")
    parser.add_argument("--num-runs", type=int, default=50, help="Number of benchmark runs")
    parser.add_argument("--warmup-runs", type=int, default=10, help="Number of warmup runs")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--compile", action="store_true", help="Test compiled optimized model too")
    
    args = parser.parse_args()
    
    # Create profiler
    profiler = ModelProfiler(args.device)
    print(f"Comparing {args.model}: optimized vs baseline")
    print("="*60)
    
    # Create models
    models = []
    
    # Baseline model
    try:
        baseline_model = create_baseline_model(args.model, args.hidden_size).to(profiler.device)
        models.append((baseline_model, f"{args.model}_baseline"))
        baseline_params = sum(p.numel() for p in baseline_model.parameters())
        print(f"Baseline {args.model}: {baseline_params:,} parameters")
    except Exception as e:
        print(f"Failed to create baseline {args.model}: {e}")
        return
    
    # Optimized model
    try:
        optimized_model = create_optimized_model(args.model, args.hidden_size).to(profiler.device)
        models.append((optimized_model, f"{args.model}_optimized"))
        opt_params = sum(p.numel() for p in optimized_model.parameters())
        print(f"Optimized {args.model}: {opt_params:,} parameters")
    except Exception as e:
        print(f"Failed to create optimized {args.model}: {e}")
        print("Make sure you have created the optimized implementation file!")
        return
    
    # Compiled model (optional)
    if args.compile and hasattr(torch, 'compile'):
        try:
            compiled_model = create_optimized_model(args.model, args.hidden_size)
            compiled_model = torch.compile(compiled_model, mode="reduce-overhead").to(profiler.device)
            models.append((compiled_model, f"{args.model}_compiled"))
            print(f"Compiled {args.model}: {opt_params:,} parameters")
        except Exception as e:
            print(f"Failed to create compiled {args.model}: {e}")
    
    # Determine input type
    if args.model == 'xlstm':
        # For xLSTM comparison, we need to handle different input types
        print(f"Using mixed input types for {args.model} comparison")
        
        # Baseline uses single timestep
        baseline_input_fn = single_timestep_input(args.batch_size, args.hidden_size, device=profiler.device)
        
        # Optimized uses tokens  
        optimized_input_fn = token_input(args.batch_size, args.seq_len, device=profiler.device)
        
        # Run separate benchmarks
        print(f"\nBaseline input: single timestep [batch_size={args.batch_size}, hidden_size={args.hidden_size}]")
        baseline_results = profiler.benchmark_model(
            baseline_model, baseline_input_fn, 
            num_runs=args.num_runs, warmup_runs=args.warmup_runs,
            name=f"{args.model}_baseline"
        )
        
        print(f"\nOptimized input: token sequence [batch_size={args.batch_size}, seq_len={args.seq_len}]")
        opt_results = profiler.benchmark_model(
            optimized_model, optimized_input_fn,
            num_runs=args.num_runs, warmup_runs=args.warmup_runs, 
            name=f"{args.model}_optimized"
        )
        
        # Per-token comparison
        baseline_per_token = baseline_results['mean_time']
        optimized_per_token = opt_results['mean_time'] / args.seq_len
        speedup = baseline_per_token / optimized_per_token
        
        print(f"\nPER-TOKEN COMPARISON:")
        print(f"Baseline:  {baseline_per_token*1000:.3f} ms/token")
        print(f"Optimized: {optimized_per_token*1000:.3f} ms/token")
        print(f"Speedup:   {speedup:.2f}x")
        
    else:
        # Other models use same input type
        input_fn = single_timestep_input(args.batch_size, args.hidden_size, device=profiler.device)
        print(f"Using single timestep input: batch_size={args.batch_size}, hidden_size={args.hidden_size}")
        
        # Run comparison
        results = profiler.compare_models(
            models, 
            input_fn,
            num_runs=args.num_runs,
            warmup_runs=args.warmup_runs
        )

if __name__ == "__main__":
    main()
