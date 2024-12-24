import time
import torch
from decimal import Decimal, ROUND_HALF_UP

GPU_TEST_ITERATIONS = 4000
GPU_TEST_SIZE = 1000

def measure_gpu_speed(device):
    """Measure the speed of a GPU by performing matrix operations."""
    start_time = time.time()
    dummy_tensor = torch.randn(GPU_TEST_SIZE, GPU_TEST_SIZE).to(device)
    for _ in range(GPU_TEST_ITERATIONS):
        _ = dummy_tensor @ dummy_tensor
    end_time = time.time()
    return 1 / (end_time - start_time)

def split_files_proportionally(filelist, speeds):
    """Split files proportionally based on GPU speeds."""
    total_speed = sum(speed for _, speed in speeds)
    proportions = [(gpu_id, speed / total_speed) for gpu_id, speed in speeds]
    chunk_sizes = [int(Decimal(len(filelist) * prop).quantize(Decimal(0), rounding=ROUND_HALF_UP)) for _, prop in proportions]

    chunks = []
    start = 0
    for gpu_id, size in zip(proportions, chunk_sizes):
        chunk = filelist[start:start + size]
        chunks.append((gpu_id[0], chunk))
        start += size

    return chunks
