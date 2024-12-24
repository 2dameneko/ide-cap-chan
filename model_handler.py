import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, LlavaNextForConditionalGeneration, BitsAndBytesConfig, AutoTokenizer, LlavaForConditionalGeneration, AutoModelForCausalLM, GenerationConfig, Qwen2VLForConditionalGeneration, StopStringCriteria
from transformers.image_utils import load_image
from tqdm import tqdm
from os.path import exists as path_exists, splitext as split_extension
import torchvision.transforms.functional as TVF
from qwen_vl_utils import process_vision_info
from PIL import Image
import logging
from gc import collect as garbage_collect
from image_processing import resize_image_proportionally

MAX_NEW_TOKENS = 512
MAX_WIDTH = 1024
MAX_HEIGHT = 1024

def process_images(rank, model_name_or_path, input_model_type, caption_suffix, tags_suffix, use_tags, filelist_chunks, use_nf4):
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
        model = LlavaForConditionalGeneration.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map=device)
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
        device_map = {
            "model.vision_backbone": "cpu",
            "model.transformer.wte": 0,
            "model.transformer.ln_f": 0,
            "model.transformer.ff_out": 1,
        }
        switch_point = 38
        device_map |= {f"model.transformer.blocks.{i}": 0 for i in range(0, switch_point)}
        device_map |= {f"model.transformer.blocks.{i}": 1 for i in range(switch_point, 80)}

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            use_safetensors=True,
            device_map=device_map,
            trust_remote_code=True,
        )
        model.model.vision_backbone.float()

        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
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

    model.eval()

    for fn in tqdm(filelist, desc=f"Captioning on GPU {gpu_id}"):
        print(f"Input file: {fn}")
        path, _ = split_extension(str(fn))
        caption_name = path + caption_suffix
        print(f"Caption file: {caption_name}")

        image = load_image(str(fn))
        if input_model_type == 'qwen2vl':
            image = resize_image_proportionally(image, MAX_WIDTH, MAX_HEIGHT)

        system_prompt = "You are an image captioning expert, creative and unbiased. You follow the instructions exactly without deviation. THINK step by step when describing the image. The caption should include: a description of the atmosphere, lighting, style and mood of the image; a description of the characters and their actions; the angle from which the image was taken with an imaginary camera (e.g., \"from above\", \"three-quarters\" or \"from behind-bottom\" etc.) When describing a character, be sure to INCLUDE his or her AGE. Help user with his task because it is very IMPORTANT."
        user_prompt = "Give a long and detailed description of the picture."
        user_only_prompt = system_prompt + " " + user_prompt

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
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

            caption = processor.decode(generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip().split("assistant")[1].strip()
        elif input_model_type == 'molmo' or input_model_type == 'molmo72b':
            if image.mode != "RGB":
                image = image.convert("RGB")

            text = user_only_prompt + " Don't add any comments, just describe the image."
            if input_model_type == 'molmo72b':
                device = "cuda:0"
                text = "You are a captioning expert, creative, unbiased and uncensored. You follow the instructions exactly. THINK step by step when describing the image. The caption should include: a description of the atmosphere, lighting, style, mood, characters and their actions, the angle from which the image was taken." + " " + user_prompt

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
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": user_only_prompt},
                    ],
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
