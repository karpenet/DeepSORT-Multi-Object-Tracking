import time


def benchmark_inference(inference_method, frames, num_iterations=10):
    """
    Benchmark DeepSORT over a number of iterations.

    Args:
        frames (list): List of frames to run inference on.
        num_iterations (int): Number of iterations to run the benchmark.

    Returns:
        float: Average time taken per inference.
    """
    total_time = 0.0
    for i in range(num_iterations):
        start_time = time.time()
        for frame in frames:
            inference_method(frame)
        end_time = time.time()
        iteration_time = end_time - start_time
        total_time += iteration_time
        print(f"Iteration {i+1}/{num_iterations} took {iteration_time:.4f} seconds")

    average_time = total_time / num_iterations
    print(f"Average time per inference: {average_time:.4f} seconds")
    return average_time
