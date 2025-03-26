import time
import statistics
import matplotlib.pyplot as plt


def benchmark_inference(inference_method, frames, num_iterations=50):
    """
    Benchmark DeepSORT over a number of iterations.

    Args:
        frames (list): List of frames to run inference on.
        num_iterations (int): Number of iterations to run the benchmark.

    Returns:
        list: List of iteration times.
    """
    print(len(frames))
    iteration_times = []
    fps_per_iteration = []

    for i in range(num_iterations):
        inference_method(frames[0])

    # for i in range(num_iterations):
    for i, frame in enumerate(frames):
        start_time = time.time()    
        inference_method(frame)
        end_time = time.time()
        iteration_time = end_time - start_time
        iteration_times.append(iteration_time)
        fps = 1 / iteration_time
        fps_per_iteration.append(fps)
        # print(f"Iteration {i+1}/{num_iterations} took {iteration_time:.4f} seconds, FPS: {fps:.2f}")

    average_time = statistics.mean(iteration_times)
    std_dev_fps = statistics.stdev(fps_per_iteration)
    average_fps = 1 / average_time

    min_fps = 1 / max(iteration_times)
    max_fps = 1 / min(iteration_times)
    print(f"Average FPS: {average_fps:.2f}")
    print(f"Standard deviation of FPS: {std_dev_fps:.2f}")
    print(f"Minimum FPS: {min_fps:.2f}")
    print(f"Maximum FPS: {max_fps:.2f}")

    # Plot FPS per iteration
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(fps_per_iteration) + 1), fps_per_iteration, marker='o')
    plt.title('FPS per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('FPS')
    plt.grid(True)
    plt.savefig('fps_per_iteration.png')
    # plt.show()

    return iteration_times
