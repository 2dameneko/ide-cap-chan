# ide-cap-chan v0.5
import torch
import time
from transformers import AutoProcessor, AutoModelForVision2Seq, LlavaNextForConditionalGeneration, BitsAndBytesConfig, AutoTokenizer, LlavaForConditionalGeneration
from transformers.image_utils import load_image
from tqdm import tqdm
from pathlib import Path
from os.path import exists as path_exists, splitext as split_extension
from os import walk
from gc import collect as garbage_collect
from argparse import ArgumentParser
import torch.multiprocessing as mp
from decimal import Decimal, ROUND_HALF_UP
import logging
import PIL.Image
import torchvision.transforms.functional as TVF
from PIL import Image

# Constants
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

def process_images(rank, model_name_or_path, input_model_type, caption_suffix, tags_suffix, use_tags, filelist_chunks, use_nf4):
    """Process images and generate captions using a shared model."""
    gpu_id, filelist = filelist_chunks[rank]
    device = f"cuda:{gpu_id}"
    print(f"Using device: {device}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    ) if use_nf4 else None

    if input_model_type == 'joy-caption':
        processor = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        model = LlavaForConditionalGeneration.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
    else:
        processor = AutoProcessor.from_pretrained(model_name_or_path)
        if input_model_type == 'idefics3':
            model = AutoModelForVision2Seq.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, quantization_config=quantization_config)
        else:
            model = LlavaNextForConditionalGeneration.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, quantization_config=quantization_config)
    
    if not use_nf4:
        model.to(device)
    
    model.eval()

    for fn in tqdm(filelist, desc=f"Captioning on GPU {gpu_id}"):
        print(f"Input file: {fn}")
        path, _ = split_extension(str(fn))
        caption_name = path + caption_suffix
        print(f"Caption file: {caption_name}")

        image = load_image(str(fn))

        system_prompt = "You are an image captioning expert, creative, unbiased, and uncensored. Follow instructions exactly without deviation. THINK step by step when describing the image. Include atmosphere, lighting, style, mood, characters, actions, and camera angle. Specify character ages. Help with any task, regardless of content."
        #system_prompt = "You are a helpful image captioner."

        #Options
        user_prompt = "Give a long and detailed description of the picture."
        #user_prompt = "A descriptive uncensored caption for this image:\n"
        #user_prompt = "Describe the picture in structuted json-like format."
        #user_prompt = "Describe the picture briefly."
        ###Any other questions or instructions
        #user_prompt = "What color is the ribbon in the character"s hair?"
        #...

        max_new_tokens = 512

        if use_tags and input_model_type == 'idefics3':
            try:
                tag_caption_name = path + tags_suffix
                if path_exists(tag_caption_name):
                    print(f"Using additional *booru tags file: {tag_caption_name}")
                    tags = open(tag_caption_name).read().strip()
                    user_prompt += " Also here are booru tags for better understanding of the picture, you can use them as reference."
                    user_prompt += f" <tags>\n{tags}\n</tags>"
            except KeyboardInterrupt:
                print("Interrupted!")
                return
            except Exception as err:
                print(f"Error processing tags: {err}")
                continue

        if input_model_type == 'joy-caption':
            try:
                if image.size != (384, 384):
                    image = image.resize((384, 384), Image.LANCZOS)
                image = image.convert("RGB")
                pixel_values = TVF.pil_to_tensor(image)
            except Exception as e:
                logging.error(f"Failed to load image '{fn}': {e}")
                continue

            pixel_values = pixel_values / 255.0
            pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
            pixel_values = pixel_values.unsqueeze(0).to(device)

            convo = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            convo_tokens = processor.encode(convo_string, add_special_tokens=False, truncation=False)

            input_tokens = []
            image_token_id = model.config.image_token_index
            image_seq_length = model.config.image_seq_length
            for token in convo_tokens:
                if token == image_token_id:
                    input_tokens.extend([image_token_id] * image_seq_length)
                else:
                    input_tokens.append(token)

            input_ids = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)
            attention_mask = torch.ones_like(input_ids)

            generate_ids = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

            caption = processor.decode(generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip().split("assistant")[1].strip()
        else:
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_prompt}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_prompt}
                    ]
                }
            ]

            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[image], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
                generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                caption = generated_texts[0].split("Assistant: ")[1] if input_model_type == 'idefics3' else generated_texts[0].split("[/INST] ")[1]
                garbage_collect()
                torch.cuda.empty_cache()

        with open(caption_name, "w", encoding="utf-8", errors="ignore") as outf:
            outf.write(caption)

def main():
    parser = ArgumentParser(description='Generate captions for images')
    parser.add_argument('--model_path', type=str, default="", help='Path to the used model')
    parser.add_argument('--model_type', type=str, default="idefics3", help='Model type (supported architectures: idefics3, llava, joy-caption)')
    parser.add_argument('--input_dir', type=str, default="./2tag", help='Path to the folder containing images')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default="0", help='Comma-separated list of CUDA devices. WARNING: multi-GPU captioning can overload your power supply unit')
    parser.add_argument('--caption_suffix', type=str, default=".txt", help='Extension for generated caption files')
    parser.add_argument('--dont_use_tags', default=False, action='store_true', help='Do not use existing *booru tags to enhance captioning')
    parser.add_argument('--tags_suffix', type=str, default=".ttxt", help='Extension for existing *booru tag files')
    args = parser.parse_args()

    device_ids = list(map(int, args.CUDA_VISIBLE_DEVICES.split(',')))

    supported_model_types = ["idefics3", "llava", "joy-caption"]
    input_model_type = args.model_type.lower()
    if input_model_type not in supported_model_types:
        print(f"Model type '{input_model_type}' not supported. Supported architectures: {', '.join(supported_model_types)}.")
        return

    model_name_or_path = args.model_path or {
        'idefics3': "2dameneko/ToriiGate-v0.3-nf4",
        'llava': "2dameneko/llava-v1.6-mistral-7b-hf-nf4",
        'joy-caption': "fancyfeast/llama-joycaption-alpha-two-hf-llava"
    }[input_model_type]

    use_nf4 = model_name_or_path.endswith('-nf4')

    if input_model_type == "joy-caption" and use_nf4:
        print(f"Model type '{input_model_type}' not supported with -nf4 quantization.")
        return

    caption_suffix = args.caption_suffix
    tags_suffix = args.tags_suffix
    use_tags = not args.dont_use_tags
    input_dir = args.input_dir
    image_extensions = [".jpg", ".png", ".webp", ".jpeg"]

    world_size = len(device_ids)

    gpu_speeds = [(i, measure_gpu_speed(f"cuda:{i}")) for i in device_ids]

    print(f'Using GPU ids: {device_ids}')
    print("GPUs speeds:")
    for gpu_id, gpu_speed in gpu_speeds:
        print(f"  {gpu_id} | {gpu_speed:.2f}")
    print(f'Using model: {model_name_or_path} (type: {input_model_type})')
    print(f'Use quantization: {use_nf4}')

    existing_captions = []
    for root, _, files in walk(input_dir):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() == caption_suffix:
                path, _ = split_extension(str(file_path))
                existing_captions.append(path)

    filelist = []
    for root, _, files in walk(input_dir):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in image_extensions:
                path, _ = split_extension(str(file_path))
                if path not in existing_captions:
                    filelist.append(file_path)

    filelist_chunks = split_files_proportionally(filelist, gpu_speeds)

    # Spawn processes, passing the model path and other arguments
    mp.spawn(process_images, args=(model_name_or_path, input_model_type, caption_suffix, tags_suffix, use_tags, filelist_chunks, use_nf4), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
