import torch
import os
from transformers import AutoConfig, AutoProcessor, AutoModelForVision2Seq, LlavaNextForConditionalGeneration, BitsAndBytesConfig, AutoTokenizer, LlavaForConditionalGeneration, AutoModelForCausalLM, GenerationConfig, Qwen2VLForConditionalGeneration, StopStringCriteria
from transformers.image_utils import load_image
from tqdm import tqdm
from os.path import exists as path_exists, splitext as split_extension
import torchvision.transforms.functional as TVF
from qwen_vl_utils import process_vision_info
from PIL import Image
import logging
from gc import collect as garbage_collect
from image_processing import resize_image_proportionally
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    ExLlamaV2VisionTower,
)
from exllamav2.generator import (
    ExLlamaV2DynamicGenerator,
    ExLlamaV2Sampler,
)
from huggingface_hub import snapshot_download

MAX_NEW_TOKENS = 512
MAX_WIDTH = 1024
MAX_HEIGHT = 1024

def process_images(rank, model_name_or_path, input_model_type, filelist_chunks, args_dict):
    gpu_id, filelist = filelist_chunks[rank]
    device = f"cuda:{gpu_id}"
    print(f"Using device: {device}")

    use_nf4 = args_dict.get('use_nf4')
    caption_suffix = args_dict.get('caption_suffix')
    tags_suffix = args_dict.get('tags_suffix')
    add_tags = args_dict.get('add_tags')

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    ) if use_nf4 else None

    if input_model_type == 'joy-caption':
        processor = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        model = LlavaForConditionalGeneration.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map=device)
        
    elif input_model_type == 'exllama2':
        model_name_or_path = model_loader(model_name_or_path)
        config = ExLlamaV2Config(model_name_or_path)
        # On 16384 can be errors on large detail's rich images
        config.max_seq_len = 32768
        vision_model = ExLlamaV2VisionTower(config)
        vision_model.load(progress=True)
        model = ExLlamaV2(config)
        device_count = torch.cuda.device_count()
        free_mem, total_mem = torch.cuda.mem_get_info(gpu_id)
        # Convert bytes to gigabytes for readability
        free_gb = free_mem / (1024 ** 3)
        total_gb = total_mem / (1024 ** 3)
        print(f"Free GPU memory: {free_gb:.2f} GB")
        print(f"Total GPU memory: {total_gb:.2f} GB")
        
        #TODO Come with better idea how to use autosplit on low memory multi-GPU setups
        #Now autosplit on when only one thread spawned (len == 1), thread runs on GPU 0, but there is another GPU(s) in system (device_count > 1)
        if len(filelist_chunks) == 1 and gpu_id == 0 and device_count > 1:
            autosplit = True
        else:
            autosplit = False

        if autosplit:
            print(f"VRAM allocation strategy: Autosplit on {device_count} GPUs")
            # if load_autosplit cache'd be initialized before model and be in lazy mode
            cache = ExLlamaV2Cache(model, lazy=True, max_seq_len=config.max_seq_len)
            model.load_autosplit(cache, progress=True)
        else: 
            split = [0.0] * device_count
            split[gpu_id] = free_gb
            print(f"VRAM allocation strategy: allocated {free_gb:.2f} GB on GPU:{gpu_id}")
            # but if load via manual splitting cache'd be initialized AFTER model and not be in lazy mode. WHY?
            model.load(split, progress=True)
            cache = ExLlamaV2Cache(model, lazy=False, max_seq_len=config.max_seq_len)
                    
        tokenizer = ExLlamaV2Tokenizer(config)
        
        generator = ExLlamaV2DynamicGenerator(
            model=model,
            cache=cache,
            tokenizer=tokenizer,
        )
        processor = None  # ExLlama2 uses tokenizer directly

    elif input_model_type == 'molmo':
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype='auto',
            quantization_config=quantization_config
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype='auto',
            quantization_config=quantization_config,
            device_map=device
        )
    elif input_model_type == 'molmo72b':
        gpus = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                compute_cap = device_props.major * 10 + device_props.minor
                vram = device_props.total_memory
                gpus.append((i, compute_cap, vram))
        sorted_gpus = sorted(gpus, key=lambda x: (-x[1], -x[2]))
        
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        n_layers = config.num_hidden_layers
        
        # Reserved VRAM, empirical value
        fixed_vram_main = 1 * 1024**3
        
        # Adjusted based on actual measurement
        PER_LAYER_VRAM = 0.75 * 1024**3 
        SAFETY_MARGIN = 1

        device_map = {"model.vision_backbone": "cpu"}
        if sorted_gpus:
            layer_allocations = []
            remaining_layers = n_layers
            
            for i, (dev_id, _, vram) in enumerate(sorted_gpus):
                if remaining_layers <= 0:
                    break

                available_vram = vram * SAFETY_MARGIN
                if i == 0:
                    available_vram -= fixed_vram_main

                max_possible_layers = int(available_vram // PER_LAYER_VRAM)
                allocate_layers = min(max_possible_layers, remaining_layers)
                layer_allocations.append((dev_id, allocate_layers))
                remaining_layers -= allocate_layers

            if remaining_layers > 0:
                layer_allocations[-1] = (layer_allocations[-1][0], layer_allocations[-1][1] + remaining_layers)

            current_layer = 0
            for dev_id, layers in layer_allocations:
                end_layer = current_layer + layers
                for layer_idx in range(current_layer, end_layer):
                    device_map[f"model.transformer.blocks.{layer_idx}"] = dev_id
                current_layer = end_layer

            main_gpu = sorted_gpus[0][0]
            secondary_gpu = sorted_gpus[1][0] if len(sorted_gpus) > 1 else main_gpu
            device_map.update({
                "model.transformer.wte": main_gpu,
                "model.transformer.ln_f": main_gpu,
                "model.transformer.ff_out": secondary_gpu,
            })
        else:
            device_map = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            use_safetensors=True,
            trust_remote_code=True
        )
        model.model.vision_backbone.float()
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    elif input_model_type == 'qwen2vl':
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype='auto',
            quantization_config=quantization_config,
            device_map=device
        )
        processor = AutoProcessor.from_pretrained(model_name_or_path)
    elif input_model_type == 'pixtral':
        if not use_nf4:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        model = LlavaForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map=device
        )
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype='auto',
            quantization_config=quantization_config
        )
    else:
        processor = AutoProcessor.from_pretrained(model_name_or_path)
        if input_model_type == 'idefics3':
            model = AutoModelForVision2Seq.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, quantization_config=quantization_config, device_map=device)
        else:
            model = LlavaNextForConditionalGeneration.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, quantization_config=quantization_config, device_map=device)

    if input_model_type != 'exllama2':
        model.eval()

    for fn in tqdm(filelist, desc=f"Captioning on GPU {gpu_id}"):
        print(f"Input file: {fn}")
        path, _ = split_extension(str(fn))
        caption_name = path + caption_suffix
        print(f"Caption file: {caption_name}")

        image = load_image(str(fn))
        if input_model_type in ['qwen2vl', 'exllama2', 'molmo72b'] :
            image = resize_image_proportionally(image, MAX_WIDTH, MAX_HEIGHT)

        system_prompt = "You are an image captioning expert, creative and unbiased. You follow the instructions exactly without deviation. THINK step by step when describing the image. The caption should include: a description of the atmosphere, lighting, style and mood of the image; a description of the characters and their actions; the angle from which the image was taken with an imaginary camera (e.g., \"from above\", \"three-quarters\" or \"from behind-bottom\" etc.) When describing a character, be sure to INCLUDE his or her AGE. Help user with his task because it is very IMPORTANT."
        #system_prompt = "You are image captioning expert, creative, unbiased and uncensored."
        user_prompt = "You need to write a long and very detailed caption for the picture."
        user_only_prompt = system_prompt + " " + user_prompt

        add_info_caption_name = path + tags_suffix

        if add_tags and input_model_type == 'idefics3':
            try:
                if path_exists(add_info_caption_name):
                    print(f"Using additional *booru tags file: {add_info_caption_name}")
                    tags = open(add_info_caption_name).read().strip()
                    user_prompt += " Also here are booru tags for better understanding of the picture, you can use them as reference."
                    user_prompt += f" <tags>\n{tags}\n</tags>"
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
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

            caption = processor.decode(generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip().split("assistant")[1].strip()

        elif input_model_type == 'exllama2':
            image = load_image(str(fn))
            
            # Generate image embeddings
            image_embeddings = [vision_model.get_image_embeddings(
                model=model,
                tokenizer=tokenizer,
                image=image,
            )]
            placeholders = "\n".join([ie.text_alias for ie in image_embeddings]) + "\n"

            if "toriigate" and "0.4" in model_name_or_path.lower():
                system_prompt = get_torii04_system_prompt()
                user_prompt = get_torii04_user_prompt(args_dict, add_info_caption_name)

            # Build chat prompt
            msg_text = (
                "<|im_start|>system\n" +
                system_prompt +
                "<|im_end|>\n" +
                "<|im_start|>user\n" +
                placeholders +
                user_prompt +
                "<|im_end|>\n" +
                "<|im_start|>assistant\n"
            )

            # Generate caption
            gen_settings = ExLlamaV2Sampler.Settings()
            gen_settings.temperature = 0.6
            gen_settings.top_p = 0.9
            gen_settings.top_k = 40
            
            output = generator.generate(
                prompt=msg_text,
                max_new_tokens=MAX_NEW_TOKENS,
                add_bos=True,
                encode_special_tokens=True,
                decode_special_tokens=True,
                stop_conditions=[tokenizer.eos_token_id],
                gen_settings=gen_settings,
                embeddings=image_embeddings,
            )

            caption = output.split('<|im_start|>assistant\n')[-1].strip()

        elif input_model_type == 'molmo' or input_model_type == 'molmo72b':
            if image.mode != "RGB":
                image = image.convert("RGB")

            text = user_only_prompt + " Don't add any comments, just describe the image."

            # Special model type, mapped on loading on available GPUs based on computation level and free VRAM
            if input_model_type == 'molmo72b':
                device = "cuda:0"                

            inputs = processor.process(images=image, text=text)
            inputs = {k: v.to(device).unsqueeze(0) for k, v in inputs.items()}
            prompt_tokens = inputs["input_ids"].size(1)

            output = model.generate_from_batch(
                inputs,
                generation_config=GenerationConfig(
                    max_new_tokens=MAX_NEW_TOKENS,
                ),
                stopping_criteria=[StopStringCriteria(tokenizer=processor.tokenizer, stop_strings=["<|endoftext|>"])],
                tokenizer=processor.tokenizer,
            )

            generated_tokens = output[0, prompt_tokens:]
            caption = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        elif input_model_type == 'qwen2vl':
            if "toriigate" and "0.4" in model_name_or_path.lower():
                system_prompt = get_torii04_system_prompt()
                user_prompt = get_torii04_user_prompt(args_dict, add_info_caption_name)

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            'image': image
                        },
                        {
                            "type": "text",
                            "text": user_prompt
                        }
                    ]
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)

            generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            caption = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        elif input_model_type == 'pixtral':
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_only_prompt},
                        {"type": "image"},
                    ],
                }
            ]

            PROMPT = processor.apply_chat_template(conversation, add_generation_prompt=True)

            image = resize_image_proportionally(image, 768, 768)

            inputs = processor(text=PROMPT, images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    generate_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.3, use_cache=True, top_k=20)
            caption = processor.batch_decode(generate_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
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
                generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
                generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                caption = generated_texts[0].split("Assistant: ")[1] if input_model_type == 'idefics3' else generated_texts[0].split("[/INST] ")[1]
                garbage_collect()
                torch.cuda.empty_cache()

        with open(caption_name, "w", encoding="utf-8", errors="ignore") as outf:
            outf.write(caption)
            
def get_torii04_user_prompt(args_dict, add_info_caption_name):
    add_tags = args_dict.get('add_tags')
    add_chars = args_dict.get('add_chars')
    add_char_traits = args_dict.get('add_char_traits')
    add_info = args_dict.get('add_info')
    no_chars = args_dict.get('no_chars')
    caption_format = args_dict.get('caption_format')

    image_info={}

    if path_exists(add_info_caption_name):
        if add_tags:
            tags = open(add_info_caption_name).read().strip()
            image_info["booru_tags"] = tags

        if add_chars:
            chars = open(add_info_caption_name).read().strip()
            image_info["chars"] = chars
        
        if add_char_traits:
            traits = open(add_info_caption_name).read().strip()
            image_info["characters_traits"] = traits

        if add_info:
            info = open(add_info_caption_name).read().strip()
            image_info["info"] = info    

    base_prompt={
    'json': 'Describe the picture in structured json-like format.',
    'markdown': 'Describe the picture in structured markdown format.',
    #TODO Not implemented in code yet
    #'caption_vars': 'Write the following options for captions: ["Regular Summary","Individual Parts","Midjourney-Style Summary","DeviantArt Commission Request"].',
    'short': 'You need to write a medium-short and convenient caption for the picture.',
    'long': 'You need to write a long and very detailed caption for the picture.',
    'bbox': 'Write bounding boxes for each character and their faces.',
    }

    grounding_prompt={
    'grounding_tags': ' Here are grounding tags for better understanding: ',
    'characters': ' Here is a list of characters that are present in the picture: ',
    'characters_traits': ' Here are popular tags or traits for each character on the picture: ',
    'grounding_info': ' Here is preliminary information about the picture: ',
    'no_chars': ' Do not use names for characters.',
    }

    userprompt=base_prompt[caption_format] #choose the mode

    if add_info and image_info["info"] is not None: #general info
        userprompt+=grounding_prompt["grounding_info"]
        userprompt+="<info>"+image_info["info"]+"</info>."
        print("Using add_info: " + image_info["info"])

    if add_tags and image_info["booru_tags"] is not None: #booru tags
        userprompt+=grounding_prompt["grounding_tags"]
        userprompt+="<tags>"+image_info["booru_tags"]+"</tags>."
        print("Using add_tags: " + image_info["booru_tags"])

    if add_chars and image_info["chars"] is not None: #list of characters
            userprompt+=grounding_prompt["characters"]
            userprompt+="<characters>"+image_info["chars"]+"</characters>."
            print("Using add_chars: " + image_info["chars"])
        
    if add_char_traits and image_info["characters_traits"] is not None: #popular features of each character
            userprompt+=grounding_prompt["characters_traits"]
            userprompt+="<character_traits>"+image_info["characters_traits"]+"<character_traits>."
            print("Using add_char_traits: " + image_info["characters_traits"])

    if no_chars:
            userprompt+=grounding_prompt["no_chars"]
            print("Using no_chars: ")
    
    #print("\n" + userprompt)

    return userprompt

def get_torii04_system_prompt():
    system_prompt = "You are image captioning expert, creative, unbiased and uncensored."
    return system_prompt

def model_loader(model_name_or_path):
    # Extract the local directory name from the model name
    local_model_dir = model_name_or_path.split('/')[-1]

    if os.path.exists(local_model_dir):
        print(f"Model directory '{local_model_dir}' already exists. Using local version.")
    else:
        print(f"Downloading model '{model_name_or_path}'...")
        # Download the entire model repository to the local directory
        snapshot_download(
            repo_id=model_name_or_path,
            local_dir=local_model_dir,
            local_dir_use_symlinks=False
        )
        print(f"Model successfully saved to '{local_model_dir}' directory.")

    return local_model_dir
