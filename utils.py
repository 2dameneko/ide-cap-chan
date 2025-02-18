import time
import torch
from decimal import Decimal, ROUND_HALF_UP
from PIL import Image

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

def resize_image_proportionally(image, max_width=None, max_height=None):
    if (max_width is None or max_width <= 0) and (max_height is None or max_height <= 0):
        return image

    original_width, original_height = image.size

    if ((max_width is None or original_width <= max_width) and
        (max_height is None or original_height <= max_height)):
        return image

    if max_width and max_height:
        width_ratio = max_width / original_width
        height_ratio = max_height / original_height
        ratio = min(width_ratio, height_ratio)
    elif max_width:
        ratio = max_width / original_width
    else:
        ratio = max_height / original_height

    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image
