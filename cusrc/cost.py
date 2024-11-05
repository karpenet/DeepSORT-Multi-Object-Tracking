import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

# Define the size of the arrays
N = 1024

# Initialize input arrays
a = np.random.randint(1, 10, N).astype(np.int32)
b = np.random.randint(1, 10, N).astype(np.int32)
c = np.zeros(N, dtype=np.int32)

# Allocate memory on the GPU
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)

# Copy data to the GPU
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

# Load and compile the CUDA kernel from the .cu file
with open('cost.cu', 'r') as f:
    kernel_code = f.read()

mod = SourceModule(kernel_code)
add = mod.get_function("compute_cost_kernel")

# Define the block and grid size
block_size = 256
grid_size = (N + block_size - 1) // block_size

# Launch the kernel
compute_cost_kernel(a_gpu, b_gpu, c_gpu, np.int32(N), block=(block_size, 1, 1), grid=(grid_size, 1))  # noqa: F821

# Copy the result back to the host
cuda.memcpy_dtoh(c, c_gpu)

# Check the result
print("Result: ", c[:10])
