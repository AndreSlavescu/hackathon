import time
import torch
import statistics
import sys
import argparse
from typing import Dict, Any


def apply_optimizations():
    """Apply all optimizations for 2.3x speedup"""
    try:
        from patching_file import apply_optimizations as patch_optimizations
        patch_optimizations()
    except ImportError:
        print("❌ patching_file.py not found - no optimizations applied")


class E2EProfiler:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        
    def benchmark_model(self, model, model_name: str, batch_size: int, num_runs: int):
        model.eval()
        
        num_features = 79
        x = torch.randn(batch_size, num_features, device=self.device)
        state = model.init_state(batch_size, self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                output, state = model(x, state)
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                output, state = model(x, state)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)
        
        num_params = sum(p.numel() for p in model.parameters())
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times)
        min_time = min(times)
        throughput = 1000 / mean_time
        
        print(f"{model_name}:")
        print(f"  Parameters: {num_params:,}")
        print(f"  Mean time: {mean_time:.3f} ± {std_time:.3f} ms")
        print(f"  Min time: {min_time:.3f} ms") 
        print(f"  Throughput: {throughput:.1f} inferences/sec")
        
        return {
            "mean_time": mean_time,
            "std_time": std_time,
            "min_time": min_time,
            "throughput": throughput,
            "num_params": num_params
        }
    
    def run_comparison(self, config, batch_size: int = 8, num_runs: int = 50):
        print(f"Device: {self.device}")
        print(f"Config: {config}")
        print(f"Batch size: {batch_size}")
        print("="*60)
        
        # Baseline model (import fresh)
        import importlib
        if 'model.inference_model' in sys.modules:
            del sys.modules['model.inference_model']
        
        from model.inference_model import MultiTowerModel
        baseline_model = MultiTowerModel(config).to(self.device)
        baseline_results = self.benchmark_model(baseline_model, "Baseline", batch_size, num_runs)
        
        # Clean up
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        del baseline_model
        
        # Apply optimizations
        apply_optimizations()
        
        # Import optimized model (should be patched now)  
        from model.inference_model import MultiTowerModel
        optimized_model = MultiTowerModel(config).to(self.device)
        optimized_results = self.benchmark_model(optimized_model, "Optimized", batch_size, num_runs)
        
        # Clean up
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        del optimized_model
        
        # Calculate and display results
        speedup = baseline_results["mean_time"] / optimized_results["mean_time"]
        throughput_improvement = optimized_results["throughput"] / baseline_results["throughput"]
        
        print("="*60)
        print("RESULTS:")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Throughput improvement: {throughput_improvement:.2f}x")
        print("="*60)
        print(f"Final speedup: {speedup:.2f}x")
        
        return {
            "baseline": baseline_results,
            "optimized": optimized_results,
            "speedup": speedup,
            "throughput_improvement": throughput_improvement
        }


def main():
    parser = argparse.ArgumentParser(description="End-to-end MultiTowerModel profiler")
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden size")
    parser.add_argument("--proj-size", type=int, default=512, help="Projection size")
    parser.add_argument("--tower-depth", type=int, default=2, help="Tower depth")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--num-features", type=int, default=79, help="Number of input features")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-runs", type=int, default=50, help="Number of benchmark runs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    from model.inference_model import ModelConfig
    config = ModelConfig(
        hidden_size=args.hidden_size,
        proj_size=args.proj_size,
        tower_depth=args.tower_depth,
        num_heads=args.num_heads,
        num_features=args.num_features
    )
    
    profiler = E2EProfiler(device=args.device)
    results = profiler.run_comparison(config, args.batch_size, args.num_runs)
    
    return results


if __name__ == "__main__":
    main()