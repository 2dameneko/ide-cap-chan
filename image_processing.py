from PIL import Image

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
