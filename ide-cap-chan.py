# ide-cap-chan v0.9minicpmo
from arg_parser import parse_arguments
from utils import measure_gpu_speed, split_files_proportionally
from image_processor import process_images
import torch.multiprocessing as mp
from os import walk
from os.path import splitext as split_extension
from pathlib import Path

def main():
    args = parse_arguments()

    device_ids = list(map(int, args.CUDA_VISIBLE_DEVICES.split(',')))

    supported_model_types = ["idefics3", "llava", "joy-caption", "molmo", "qwen2vl", "molmo72b", "pixtral", "exllama2", "minicpmo", "generic"]
    input_model_type = args.model_type.lower()
    if input_model_type not in supported_model_types:
        print(f"Model type '{input_model_type}' not supported. Supported loaders: {', '.join(supported_model_types)}.")
        return

    model_name_or_path = args.model_path or {
        'idefics3': "2dameneko/ToriiGate-v0.3-nf4",
        'llava': "2dameneko/llava-v1.6-mistral-7b-hf-nf4",
        'joy-caption': "fancyfeast/llama-joycaption-alpha-two-hf-llava",
        #'qwen2vl': "Ertugrul/Qwen2-VL-7B-Captioner-Relaxed",
        #'qwen2vl': "Minthy/ToriiGate-v0.4-2B",        
        'qwen2vl': "Vikhrmodels/Vikhr-2-VL-2b-Instruct-experimental",
        'molmo': "cyan2k/molmo-7B-O-bnb-4bit",
        'molmo72b': "SeanScripts/Molmo-72B-0924-nf4",
        'pixtral': "Ertugrul/Pixtral-12B-Captioner-Relaxed",
        'exllama2': "Minthy/ToriiGate-v0.4-2B-exl2-8bpw",
        #'exllama2': "Minthy/ToriiGate-v0.4-7B-exl2-8bpw",
        'minicpmo': "openbmb/MiniCPM-o-2_6",
        'generic': None,
            }[input_model_type]

    quant_suffixes = ["nf4", "bnb", "4bit"]
    use_nf4 = any(suffix in model_name_or_path for suffix in quant_suffixes)

    if input_model_type == "joy-caption" and use_nf4:
        print(f"Model type '{input_model_type}' not supported with -nf4 quantization. Set to false.")
        use_nf4 = False

    args_dict = {
        'use_nf4' : use_nf4,
        'caption_suffix' : args.caption_suffix,             
        'tags_suffix' : args.tags_suffix,
        'add_tags' : args.add_tags,
        'add_chars': args.add_chars,
        'add_char_traits' : args.add_char_traits,
        'add_info' : args.add_info,
        'no_chars' : args.no_chars,
        'caption_format' : args.caption_format,
    }

    input_dir = args.input_dir
    image_extensions = [".jpg", ".png", ".webp", ".jpeg"]

    world_size = len(device_ids)

    gpu_speeds = [(i, measure_gpu_speed(f"cuda:{i}")) for i in device_ids]

    print(f'Using GPU ids: {device_ids}')
    print("GPUs speeds:")
    for gpu_id, gpu_speed in gpu_speeds:
        print(f"  {gpu_id} | {gpu_speed:.2f}")
    print(f'Using model: {model_name_or_path} (type: {input_model_type})')
    print(f'Use quantization: {args_dict.get("use_nf4")}')

    existing_captions = []
    for root, _, files in walk(input_dir):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() == args_dict.get('caption_suffix'):
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
    mp.spawn(process_images, args=(model_name_or_path, input_model_type, filelist_chunks, args_dict), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
