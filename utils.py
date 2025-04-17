import time
import torch
from PIL import Image

GPU_TEST_ITERATIONS = 4000
GPU_TEST_SIZE = 1000

def measure_gpu_speed(device):
    """
    Measure the speed of a GPU by performing matrix operations.
    
    Args:
        device: The CUDA device to measure
        
    Returns:
        float: A score representing the relative speed of the GPU
    """
    start_time = time.time()
    dummy_tensor = torch.randn(GPU_TEST_SIZE, GPU_TEST_SIZE).to(device)
    for _ in range(GPU_TEST_ITERATIONS):
        _ = dummy_tensor @ dummy_tensor
    end_time = time.time()
    return 1 / (end_time - start_time)

def resize_image_proportionally(image, max_width=None, max_height=None):
    """
    Resize an image proportionally to fit within the specified dimensions.
    
    Args:
        image: PIL Image to resize
        max_width: Maximum width
        max_height: Maximum height
        
    Returns:
        PIL Image: Resized image
    """
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
