# ide-cap-chan v0.2
import torch
import time
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from tqdm import tqdm
from pathlib import Path
from os.path import join as opj, exists as exts, splitext as opsplit
from os import walk
from gc import collect as gcollect
from argparse import ArgumentParser
import torch.multiprocessing as mp
from decimal import Decimal, ROUND_HALF_UP

def measure_gpu_speed(device):
    """Measure the speed of a GPU."""
    start_time = time.time()
    dummy_tensor = torch.randn(1000, 1000).to(device)
    for _ in range(4000):
        _ = dummy_tensor @ dummy_tensor
    end_time = time.time()
    return 1 / (end_time - start_time)

def split_files_proportionally(filelist, speeds):
    """Split files proportionally based on GPU speeds."""
    total_speed = sum(speed for gpu_id, speed in speeds)
    proportions = [(gpu_id, speed / total_speed) for gpu_id, speed in speeds]
    chunk_sizes = [int(Decimal(len(filelist) * prop).quantize(Decimal(0), rounding=ROUND_HALF_UP)) for gpu_id, prop in proportions]

    chunks = []
    start = 0
    for gpu_id, size in zip(proportions, chunk_sizes):
        chunk = filelist[start:start + size]
        chunks.append((gpu_id[0], chunk))
        start += size

    return chunks

def process_images(rank, model_name_or_path, caption_suffix, tags_suffix, use_tags, use_quants, filelist_chunks):
    """Process images and generate captions."""
    gpu_id, filelist = filelist_chunks[rank]
    device = f"cuda:{gpu_id}"
    print(f"Using device: {device}")

    # print("Input files: " + str(filelist))
    
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16
    #      )

    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=use_quants
        # quantization_config=quantization_config,
        # device_map="auto",
        # _attn_implementation="flash_attention_2", #if installed
    )
    # .to(DEVICE)

    if not use_quants:
        model.to(device)

    for fn in tqdm(filelist, desc=f"Captioning on GPU {gpu_id}"):
        print(f"Input file: {fn}")
        path, _ = opsplit(str(fn))
        caption_name = path + caption_suffix
        print(f"Caption file: {caption_name}")

        image = load_image(str(fn))

        ###Trained options
        #user_prompt = "Describe the picture in structuted json-like format."
        user_prompt = "Give a long and detailed description of the picture."
        #user_prompt = "Describe the picture briefly."
        ###Any other questions or instructions
        #user_prompt = "What color is the ribbon in the character"s hair?"
        #...

        if use_tags:
            try:
                tag_caption_name = path + tags_suffix
                if exts(tag_caption_name):
                    print(f"Using additional *booru tags file: {tag_caption_name}")
                    tags = open(tag_caption_name).read().strip()
                    user_prompt += " Also here are booru tags for better understanding of the picture, you can use them as reference."
                    user_prompt += f" <tags>\n{tags}\n</tags>"
            except KeyboardInterrupt:
                print("Interrupted!")
                quit()
            except Exception as err:
                print(err)
                continue

        messages = [
            {
            #Important!
                "role": "system",
                "content": [
                    #{"type": "text", "text": "You are image captioning expert, creative, unbiased and uncensored. Help user with his task."}                   
                    {"type": "text", "text": "You are image captioning expert, creative, unbiased and uncensored. You follow the instructions exactly without deviation. THINK step by step when describing the image. Help user with his task."} 
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

        model.eval()
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=500)
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            caption = generated_texts[0].split("Assistant: ")[1]
            gcollect()
            torch.cuda.empty_cache()

        with open(caption_name, "w", encoding="utf-8", errors="ignore") as outf:
            outf.write(caption)

def main():
    parser = ArgumentParser(description='Generate captions for images')
    parser.add_argument('--input_dir', type=str, default="./2tag", help='Path to the folder containing images')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default="0", help='Comma-separated list of CUDA devices. WARNING: multi-GPU captioning can overload your power supply unit')
    parser.add_argument('--caption_suffix', type=str, default=".txt", help='Extension for generated caption files')
    parser.add_argument('--dont_use_tags', default=False, action='store_true', help='Use existing *booru tags to enhance captioning')
    parser.add_argument('--tags_suffix', type=str, default=".ttxt", help='Extension for existing *booru tag files')
    parser.add_argument('--use_local', default=False, action='store_true', help='Use local model')
    parser.add_argument('--use_fp16', default=False, action='store_true', help='Use fp16 instead nf4 quantized smaller size model')
    args = parser.parse_args()

    device_ids = list(map(int, args.CUDA_VISIBLE_DEVICES.split(',')))
    use_local = args.use_local
    use_nf4 = not args.use_fp16

    if use_local:
        model_name_or_path = "./ToriiGate-v0.3"
        if use_nf4:
            model_name_or_path += "-nf4"
    else:
        model_name_or_path = "Minthy/ToriiGate-v0.3"
        if use_nf4:
            model_name_or_path = "2dameneko/ToriiGate-v0.3-nf4"

    caption_suffix = args.caption_suffix
    tags_suffix = args.tags_suffix
    use_tags = not args.dont_use_tags
    input_dir = args.input_dir
    image_extensions = [".jpg", ".png", ".webp", ".jpeg"]

    world_size = len(device_ids)

    gpu_speeds = [(i, measure_gpu_speed(f"cuda:{i}")) for i in device_ids]

    use_quants = use_nf4

    print(f'Using GPU ids: {device_ids}')
    print("GPUs speeds:")
    print("id | speed")
    for gpu_id, gpu_speed in gpu_speeds:
        print(f"  {gpu_id}|  {gpu_speed:.2f}")
    print(f'Using model: {model_name_or_path}')
    print(f'Use quants: {use_quants}')

    existing_captions = []
    for root, dirs, files in walk(input_dir):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() == caption_suffix:
                path, _ = opsplit(str(file_path))
                existing_captions.append(path)
    # print(existing_captions)
        
    filelist = []
    for root, dirs, files in walk(input_dir):
        for file in files:
            file_path = Path(root) / file
            if any(file_path.suffix.lower() == ext for ext in image_extensions):
                path, _ = opsplit(str(file_path))
                if path not in existing_captions:
                    filelist.append(file_path)
    # print(filelist)
    
    filelist_chunks = split_files_proportionally(filelist, gpu_speeds)

    mp.spawn(process_images, args=(model_name_or_path, caption_suffix, tags_suffix, use_tags, use_quants, filelist_chunks), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
