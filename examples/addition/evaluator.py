"""
Evaluator for Triton Vector Addition Kernel

This evaluator tests the vector addition kernel for:
1. Correctness - ensuring accurate results
2. Performance - measuring execution speed
3. Robustness - handling edge cases and different input sizes
4. Memory efficiency - tracking GPU memory usage
"""

import importlib.util
import numpy as np
import time
import traceback
import sys
import os
import torch
import triton
import psutil
import gc
from typing import Dict, Any, Tuple, Optional
import concurrent.futures


class VectorAdditionEvaluator:
    """Comprehensive evaluator for Triton vector addition kernels"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.baseline_times = {}
        self.error_count = 0
        self.success_count = 0
        
    def evaluate(self, program_path: str) -> Dict[str, Any]:
        """
        Main evaluation function for the vector addition kernel
        
        Args:
            program_path: Path to the program file
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            # Load the program
            spec = importlib.util.spec_from_file_location("program", program_path)
            program = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(program)
            
            # Check if required functions exist
            if not hasattr(program, 'vector_add'):
                return self._create_error_result("Missing vector_add function")
            
            if not hasattr(program, 'add_kernel'):
                return self._create_error_result("Missing add_kernel function")
            
            # Run comprehensive evaluation
            results = {}
            
            # Test 1: Basic correctness
            correctness_result = self._test_correctness(program)
            results.update(correctness_result)
            
            # Test 2: Performance benchmarking
            performance_result = self._test_performance(program)
            results.update(performance_result)
            
            # Test 3: Edge cases and robustness
            robustness_result = self._test_robustness(program)
            results.update(robustness_result)
            
            # Test 4: Memory efficiency
            memory_result = self._test_memory_efficiency(program)
            results.update(memory_result)
            
            # Calculate combined score
            combined_score = self._calculate_combined_score(results)
            results['combined_score'] = combined_score
            
            return results
            
        except Exception as e:
            return self._create_error_result(f"Evaluation failed: {str(e)}")
    
    def _test_correctness(self, program) -> Dict[str, float]:
        """Test basic correctness of the vector addition"""
        try:
            # Test cases with different sizes
            test_cases = [
                (1000, torch.float32),
                (10000, torch.float32),
                (100000, torch.float32),
                (1000, torch.float16),
                (10000, torch.float16),
            ]
            
            total_accuracy = 0.0
            num_tests = 0
            
            for size, dtype in test_cases:
                # Generate test data
                a = torch.randn(size, dtype=dtype, device=self.device)
                b = torch.randn(size, dtype=dtype, device=self.device)
                expected = a + b
                
                # Run kernel
                try:
                    result = program.vector_add(a, b)
                    
                    # Check accuracy
                    if dtype == torch.float16:
                        # Lower precision tolerance for float16
                        tolerance = 1e-3
                    else:
                        tolerance = 1e-6
                    
                    accuracy = torch.allclose(result, expected, rtol=tolerance, atol=tolerance)
                    total_accuracy += float(accuracy)
                    num_tests += 1
                    
                except Exception as e:
                    print(f"Correctness test failed for size {size}, dtype {dtype}: {e}")
                    continue
            
            accuracy_score = total_accuracy / max(num_tests, 1)
            return {'correctness_score': accuracy_score}
            
        except Exception as e:
            print(f"Correctness test error: {e}")
            return {'correctness_score': 0.0}
    
    def _test_performance(self, program) -> Dict[str, float]:
        """Test performance across different input sizes"""
        try:
            # Performance test cases
            test_sizes = [1000, 10000, 100000, 1000000]
            num_trials = 5
            
            baseline_times = {}
            kernel_times = {}
            
            for size in test_sizes:
                # Generate test data
                a = torch.randn(size, dtype=torch.float32, device=self.device)
                b = torch.randn(size, dtype=torch.float32, device=self.device)
                
                # Warm up
                for _ in range(3):
                    _ = program.vector_add(a, b)
                
                # Benchmark baseline (PyTorch)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(num_trials):
                    baseline_result = a + b
                torch.cuda.synchronize()
                baseline_time = (time.time() - start_time) / num_trials
                baseline_times[size] = baseline_time
                
                # Benchmark kernel
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(num_trials):
                    kernel_result = program.vector_add(a, b)
                torch.cuda.synchronize()
                kernel_time = (time.time() - start_time) / num_trials
                kernel_times[size] = kernel_time
            
            # Calculate speedup scores
            speedup_scores = []
            for size in test_sizes:
                if size in baseline_times and size in kernel_times:
                    speedup = baseline_times[size] / max(kernel_times[size], 1e-9)
                    speedup_scores.append(min(speedup, 10.0))  # Cap at 10x speedup
            
            avg_speedup = np.mean(speedup_scores) if speedup_scores else 1.0
            speed_score = min(avg_speedup / 5.0, 1.0)  # Normalize to 0-1
            
            return {
                'speed_score': speed_score,
                'avg_speedup': avg_speedup,
                'baseline_times': baseline_times,
                'kernel_times': kernel_times
            }
            
        except Exception as e:
            print(f"Performance test error: {e}")
            return {'speed_score': 0.0, 'avg_speedup': 1.0}
    
    def _test_robustness(self, program) -> Dict[str, float]:
        """Test robustness with edge cases"""
        try:
            robustness_score = 0.0
            total_tests = 0
            
            # Test 1: Zero vectors
            try:
                a = torch.zeros(1000, device=self.device)
                b = torch.zeros(1000, device=self.device)
                result = program.vector_add(a, b)
                expected = a + b
                if torch.allclose(result, expected):
                    robustness_score += 1.0
                total_tests += 1
            except Exception as e:
                print(f"Zero vector test failed: {e}")
            
            # Test 2: Large vectors
            try:
                a = torch.randn(1000000, device=self.device)
                b = torch.randn(1000000, device=self.device)
                result = program.vector_add(a, b)
                expected = a + b
                if torch.allclose(result, expected):
                    robustness_score += 1.0
                total_tests += 1
            except Exception as e:
                print(f"Large vector test failed: {e}")
            
            # Test 3: Different data types
            try:
                a = torch.randn(1000, dtype=torch.float16, device=self.device)
                b = torch.randn(1000, dtype=torch.float16, device=self.device)
                result = program.vector_add(a, b)
                expected = a + b
                if torch.allclose(result, expected, rtol=1e-3):
                    robustness_score += 1.0
                total_tests += 1
            except Exception as e:
                print(f"Float16 test failed: {e}")
            
            # Test 4: Negative numbers
            try:
                a = torch.randn(1000, device=self.device) * -1
                b = torch.randn(1000, device=self.device) * -1
                result = program.vector_add(a, b)
                expected = a + b
                if torch.allclose(result, expected):
                    robustness_score += 1.0
                total_tests += 1
            except Exception as e:
                print(f"Negative numbers test failed: {e}")
            
            final_robustness = robustness_score / max(total_tests, 1)
            return {'robustness_score': final_robustness}
            
        except Exception as e:
            print(f"Robustness test error: {e}")
            return {'robustness_score': 0.0}
    
    def _test_memory_efficiency(self, program) -> Dict[str, float]:
        """Test memory efficiency"""
        try:
            # Get initial memory state
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()
            else:
                initial_memory = psutil.virtual_memory().used
            
            # Test with large vectors
            size = 1000000
            a = torch.randn(size, device=self.device)
            b = torch.randn(size, device=self.device)
            
            # Measure memory usage
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                before_memory = torch.cuda.memory_allocated()
                result = program.vector_add(a, b)
                torch.cuda.synchronize()
                after_memory = torch.cuda.memory_allocated()
                peak_memory = torch.cuda.max_memory_allocated()
            else:
                before_memory = psutil.virtual_memory().used
                result = program.vector_add(a, b)
                after_memory = psutil.virtual_memory().used
                peak_memory = after_memory
            
            # Calculate memory efficiency
            memory_used = after_memory - before_memory
            expected_memory = a.element_size() * size * 3  # a, b, result
            
            if memory_used <= expected_memory * 1.5:  # Allow 50% overhead
                memory_score = 1.0
            else:
                memory_score = max(0.0, 1.0 - (memory_used - expected_memory) / expected_memory)
            
            return {
                'memory_score': memory_score,
                'memory_used_mb': memory_used / (1024 * 1024),
                'expected_memory_mb': expected_memory / (1024 * 1024)
            }
            
        except Exception as e:
            print(f"Memory efficiency test error: {e}")
            return {'memory_score': 0.0}
    
    def _calculate_combined_score(self, results: Dict[str, Any]) -> float:
        """Calculate combined score from all metrics"""
        weights = {
            'correctness_score': 0.4,
            'speed_score': 0.3,
            'robustness_score': 0.2,
            'memory_score': 0.1
        }
        
        combined_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in results:
                combined_score += results[metric] * weight
                total_weight += weight
        
        if total_weight > 0:
            return combined_score / total_weight
        else:
            return 0.0
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result dictionary"""
        return {
            'correctness_score': 0.0,
            'speed_score': 0.0,
            'robustness_score': 0.0,
            'memory_score': 0.0,
            'combined_score': 0.0,
            'error': error_message
        }


def evaluate(program_path: str) -> Dict[str, Any]:
    """
    Main evaluation function for OpenEvolve compatibility
    
    Args:
        program_path: Path to the program file
        
    Returns:
        Dictionary of evaluation metrics
    """
    evaluator = VectorAdditionEvaluator()
    return evaluator.evaluate(program_path)


def evaluate_stage1(program_path: str) -> Dict[str, Any]:
    """
    Stage 1 evaluation - basic correctness and functionality
    
    Args:
        program_path: Path to the program file
        
    Returns:
        Dictionary of stage 1 metrics
    """
    evaluator = VectorAdditionEvaluator()
    
    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # Check basic functionality
        if not hasattr(program, 'vector_add'):
            return {'stage1_score': 0.0, 'error': 'Missing vector_add function'}
        
        if not hasattr(program, 'add_kernel'):
            return {'stage1_score': 0.0, 'error': 'Missing add_kernel function'}
        
        # Test basic correctness
        correctness_result = evaluator._test_correctness(program)
        stage1_score = correctness_result.get('correctness_score', 0.0)
        
        return {
            'stage1_score': stage1_score,
            'correctness_score': stage1_score
        }
        
    except Exception as e:
        return {'stage1_score': 0.0, 'error': f'Stage 1 evaluation failed: {str(e)}'}


def evaluate_stage2(program_path: str) -> Dict[str, Any]:
    """
    Stage 2 evaluation - performance and robustness
    
    Args:
        program_path: Path to the program file
        
    Returns:
        Dictionary of stage 2 metrics
    """
    evaluator = VectorAdditionEvaluator()
    
    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # Test performance and robustness
        performance_result = evaluator._test_performance(program)
        robustness_result = evaluator._test_robustness(program)
        
        stage2_score = (
            performance_result.get('speed_score', 0.0) * 0.6 +
            robustness_result.get('robustness_score', 0.0) * 0.4
        )
        
        return {
            'stage2_score': stage2_score,
            'speed_score': performance_result.get('speed_score', 0.0),
            'robustness_score': robustness_result.get('robustness_score', 0.0),
            'avg_speedup': performance_result.get('avg_speedup', 1.0)
        }
        
    except Exception as e:
        return {'stage2_score': 0.0, 'error': f'Stage 2 evaluation failed: {str(e)}'}


if __name__ == "__main__":
    # Test the evaluator
    evaluator = VectorAdditionEvaluator()
    result = evaluator.evaluate("initial_program.py")
    print("Evaluation Results:")
    for key, value in result.items():
        print(f"  {key}: {value}") 