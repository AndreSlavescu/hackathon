import time
import torch
import statistics
import sys
import argparse
from typing import Dict, Any

def patch_create_layer():
    import model.inference_model as inf_model
    from optimized_xlstm_layer import create_optimized_xlstm_layer
    
    original_create_layer = inf_model.create_layer
    
    def optimized_create_layer(layer_type, hidden_size: int, num_heads: int = 8):
        if layer_type == inf_model.LayerType.XLSTM:
            return create_optimized_xlstm_layer(hidden_size, num_heads)
        else:
            return original_create_layer(layer_type, hidden_size, num_heads)
    
    inf_model.create_layer = optimized_create_layer
    return original_create_layer

def restore_create_layer(original_func):
    import model.inference_model as inf_model
    inf_model.create_layer = original_func
    modules_to_remove = [k for k in sys.modules.keys() if 'inference_model' in k]
    for module in modules_to_remove:
        if module in sys.modules:
            del sys.modules[module]

class E2EProfiler:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        
    def create_model(self, config, use_optimized: bool = False):
        if use_optimized:
            original_func = patch_create_layer()
        
        try:
            from model.inference_model import MultiTowerModel
            model = MultiTowerModel(config).to(self.device)
            return model, original_func if use_optimized else None
        except Exception as e:
            if use_optimized:
                restore_create_layer(original_func)
            raise e
    
    def benchmark_model(self, model, model_name: str, batch_size: int, num_runs: int):
        model.eval()
        
        config = model.towers[0].blocks[0].layer.__dict__.get('hidden_size') or 256
        num_features = 79
        
        x = torch.randn(batch_size, num_features, device=self.device)
        state = model.init_state(batch_size, self.device)
        
        with torch.no_grad():
            for _ in range(10):
                output, state = model(x, state)
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
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
        
        baseline_model, _ = self.create_model(config, use_optimized=False)
        baseline_results = self.benchmark_model(baseline_model, "Baseline", batch_size, num_runs)
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        del baseline_model
        
        optimized_model, original_func = self.create_model(config, use_optimized=True)
        try:
            optimized_results = self.benchmark_model(optimized_model, "Optimized", batch_size, num_runs)
        finally:
            restore_create_layer(original_func)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            del optimized_model
        
        speedup = baseline_results["mean_time"] / optimized_results["mean_time"]
        throughput_improvement = optimized_results["throughput"] / baseline_results["throughput"]
        
        print("="*60)
        print("RESULTS:")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Throughput improvement: {throughput_improvement:.2f}x")
        print("="*60)
        
        return {
            "baseline": baseline_results,
            "optimized": optimized_results,
            "speedup": speedup,
            "throughput_improvement": throughput_improvement
        }

def main():
    parser = argparse.ArgumentParser(description="End-to-end MultiTowerModel profiler")
    parser.add_argument("--hidden-size", type=int, default=512, help="Hidden size")
    parser.add_argument("--tower-depth", type=int, default=4, help="Tower depth")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-runs", type=int, default=50, help="Number of benchmark runs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    from model.inference_model import ModelConfig
    config = ModelConfig(
        hidden_size=args.hidden_size,
        proj_size=args.hidden_size * 2,
        tower_depth=args.tower_depth,
        num_heads=args.num_heads,
        num_features=79,
    )
    
    profiler = E2EProfiler(args.device)
    results = profiler.run_comparison(config, args.batch_size, args.num_runs)
    
    print(f"Final speedup: {results['speedup']:.2f}x")

if __name__ == "__main__":
    main()
