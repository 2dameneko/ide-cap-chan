from typing import Dict, Any
from abc import ABC, abstractmethod
import torch
from PIL import Image
import os
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
    GenerationConfig,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    Qwen2VLForConditionalGeneration,
    StopStringCriteria    
)
from transformers.image_utils import load_image
import torchvision.transforms.functional as TVF
from qwen_vl_utils import process_vision_info
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    ExLlamaV2VisionTower,
)
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler
from utils import resize_image_proportionally
from huggingface_hub import snapshot_download

class ModelHandler(ABC):
    def __init__(self, model_name_or_path: str, device: str, args_dict: Dict[str, Any]):
        self.model_name_or_path = self.model_loader(model_name_or_path)
        self.device = device
        self.args_dict = args_dict
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.quantization_config = None
        self._initialize_model()

    @abstractmethod
    def _initialize_model(self):
        pass

    @abstractmethod
    def process_image(self, system_prompt: str, user_prompt: str, image: Image.Image) -> str:
        pass

    def save_caption(self, caption: str, caption_path: str, encoding: str = "utf-8", errors: str = "ignore"):
        with open(caption_path, "w", encoding=encoding, errors=errors) as outf:
            outf.write(caption)

    def model_loader(self, model_name_or_path: str) -> str:
        local_model_dir = model_name_or_path.split('/')[-1]
        if os.path.exists(local_model_dir):
            print(f"Model directory '{local_model_dir}' already exists. Using local version.")
            return local_model_dir
        else:
            print(f"Downloading model '{model_name_or_path}'...")
            snapshot_download(
                repo_id=model_name_or_path,
                local_dir=local_model_dir,
            )
            print(f"Model successfully saved to '{local_model_dir}' directory.")
            return local_model_dir
    
    def _get_quantization_config(self):
        use_nf4 = self.args_dict.get('use_nf4')
        if use_nf4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        return None

class ExLLaMA2Handler(ModelHandler):
    def __init__(self, model_name_or_path: str, device: str, args_dict: Dict[str, Any]):
        super().__init__(model_name_or_path, device, args_dict)

    def _initialize_model(self):
        self.config = ExLlamaV2Config(self.model_name_or_path)
        self.config.max_seq_len = 32768

        self.vision_model = ExLlamaV2VisionTower(self.config)
        self.vision_model.load(progress=True)

        self.model = ExLlamaV2(self.config)
        self.tokenizer = ExLlamaV2Tokenizer(self.config)

        device_count = torch.cuda.device_count()
        free_mem, total_mem = torch.cuda.mem_get_info(self.device)
        free_gb = free_mem / (1024 ** 3)
        total_gb = total_mem / (1024 ** 3)

        if len(self.args_dict.get('filelist_chunks', [])) == 1 and self.device == "cuda:0" and device_count > 1:
            autosplit = True
        else:
            autosplit = False

        if autosplit:
            print(f"VRAM allocation strategy: Autosplit on {device_count} GPUs")
            cache = ExLlamaV2Cache(self.model, lazy=True, max_seq_len=self.config.max_seq_len)
            self.model.load_autosplit(cache, progress=True)
        else:
            split = [0.0] * device_count
            gpu_id = int(self.device.split(":")[1])
            split[gpu_id] = free_gb
            print(f"VRAM allocation strategy: allocated {free_gb:.2f} GB on GPU:{gpu_id}")
            self.model.load(split, progress=True)
            cache = ExLlamaV2Cache(self.model, lazy=False, max_seq_len=self.config.max_seq_len)

        self.generator = ExLlamaV2DynamicGenerator(
            model=self.model,
            cache=cache,
            tokenizer=self.tokenizer,
        )

    def model_loader(self, model_name_or_path: str) -> str:
        local_model_dir = model_name_or_path.split('/')[-1]
        if os.path.exists(local_model_dir):
            print(f"Model directory '{local_model_dir}' already exists. Using local version.")
            return local_model_dir
        else:
            print(f"Downloading model '{model_name_or_path}'...")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=model_name_or_path,
                local_dir=local_model_dir,
                local_dir_use_symlinks=False
            )
            print(f"Model successfully saved to '{local_model_dir}' directory.")
            return local_model_dir

    def process_image(self,  system_prompt: str, user_prompt: str, image: Image.Image) -> str:
        image = resize_image_proportionally(image, 2048, 2048)
        image_embeddings = [self.vision_model.get_image_embeddings(
            model=self.model,
            tokenizer=self.tokenizer,
            image=image,
        )]
        placeholders = "\n".join([ie.text_alias for ie in image_embeddings]) + "\n"

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

        gen_settings = ExLlamaV2Sampler.Settings()
        gen_settings.temperature = 0.6
        gen_settings.top_p = 0.9
        gen_settings.top_k = 40

        output = self.generator.generate(
            prompt=msg_text,
            max_new_tokens=512,
            add_bos=True,
            encode_special_tokens=True,
            decode_special_tokens=True,
            stop_conditions=[self.tokenizer.eos_token_id],
            gen_settings=gen_settings,
            embeddings=image_embeddings,
        )

        return output.split('<|im_start|>assistant\n')[-1].strip()

class JoyCaptionHandler(ModelHandler):
    def __init__(self, model_name_or_path: str, device: str, args_dict: Dict[str, Any]):
        super().__init__(model_name_or_path, device, args_dict)

    def _initialize_model(self):
        self.quantization_config = self._get_quantization_config()
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.bfloat16,
            quantization_config=self.quantization_config,
            device_map=self.device
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=self.quantization_config,
            use_fast=False
        )
        self.model.eval()

    def process_image(self,  system_prompt: str, user_prompt: str, image: Image.Image) -> str:
        if image.size != (384, 384):
            image = image.resize((384, 384), Image.LANCZOS)
        pixel_values = TVF.pil_to_tensor(image)
        pixel_values = (pixel_values / 255.0)
        pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
        pixel_values = pixel_values.unsqueeze(0).to(self.device)

        convo = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        convo_string = self.processor.apply_chat_template(convo, tokenize = False, add_generation_prompt = True)

        inputs = self.processor(text=[convo_string], images=[image], return_tensors="pt").to(self.device)
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

        generate_ids = self.model.generate(
            		**inputs,
                    max_new_tokens=300,
		            do_sample=True,
		            suppress_tokens=None,
		            use_cache=True,
		            temperature=0.6,
		            top_k=None,
		            top_p=0.9,
            	)[0]
        
        generate_ids = generate_ids[inputs['input_ids'].shape[1]:]

        caption = self.processor.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        caption = caption.strip()

        return caption

class MoLMoHandler(ModelHandler):
    def __init__(self, model_name_or_path: str, device: str, args_dict: Dict[str, Any]):
        super().__init__(model_name_or_path, device, args_dict)
        
    def _initialize_model(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            torch_dtype='auto',
            use_fast=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            torch_dtype='auto',
            attn_implementation='eager', # sdpa or flash_attention_2 or "eager"
            device_map=self.device
        )
        self.model.eval()

    def process_image(self,  system_prompt: str, user_prompt: str, image: Image.Image) -> str:
        image = resize_image_proportionally(image, 1024, 1024)

        user_only_prompt = system_prompt + " " + user_prompt

        inputs = self.processor.process(images=image, text=user_only_prompt)
        inputs = {k: v.to(self.device).unsqueeze(0) for k, v in inputs.items()}
        prompt_tokens = inputs["input_ids"].size(1)

        output = self.model.generate_from_batch(
            inputs,
            generation_config=GenerationConfig(
                max_new_tokens=512,
            ),
            stopping_criteria=[StopStringCriteria(tokenizer=self.processor.tokenizer, stop_strings=["<|endoftext|>"])],
            tokenizer=self.processor.tokenizer,
        )

        generated_tokens = output[0, prompt_tokens:]
        caption = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return caption

class MoLMo72bHandler(ModelHandler):
    def __init__(self, model_name_or_path: str, device: str, args_dict: Dict[str, Any]):
        super().__init__(model_name_or_path, device, args_dict)

    def _initialize_model(self):
        gpus = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                compute_cap = device_props.major * 10 + device_props.minor
                vram = device_props.total_memory
                gpus.append((i, compute_cap, vram))

        sorted_gpus = sorted(gpus, key=lambda x: (-x[1], -x[2]))

        config = AutoConfig.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        n_layers = config.num_hidden_layers

        fixed_vram_main = 1 * 1024**3
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

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            device_map=device_map,
            attn_implementation='eager', # sdpa or flash_attention_2 or "eager"
            torch_dtype=torch.bfloat16,            
            use_safetensors=True,
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path, trust_remote_code=True, use_fast=False)
        self.model.model.vision_backbone.float()

    def process_image(self, system_prompt: str, user_prompt: str, image: Image.Image) -> str:
        image = resize_image_proportionally(image, 1024, 1024)

        user_only_prompt = system_prompt + " " + user_prompt

        inputs = self.processor.process(images=image, text=user_only_prompt)
        inputs = {k: v.to(self.device).unsqueeze(0) for k, v in inputs.items()}
        prompt_tokens = inputs["input_ids"].size(1)

        output = self.model.generate_from_batch(
            inputs,
            generation_config=GenerationConfig(
                max_new_tokens=512,
            ),
            stopping_criteria=[StopStringCriteria(tokenizer=self.processor.tokenizer, stop_strings=["<|qqqq|>"])],
            tokenizer=self.processor.tokenizer,
        )

        generated_tokens = output[0, prompt_tokens:]
        caption = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return caption

class Qwen2VLHandler(ModelHandler):
    def __init__(self, model_name_or_path: str, device: str, args_dict: Dict[str, Any]):
        super().__init__(model_name_or_path, device, args_dict)

    def _initialize_model(self):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            torch_dtype='auto',
            quantization_config=self._get_quantization_config(),
            device_map=self.device
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)

    def process_image(self, system_prompt: str, user_prompt: str, image: Image.Image) -> str:
        image = resize_image_proportionally(image, 1024, 1024)
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

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        caption = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return caption

class PixtralHandler(ModelHandler):
    def __init__(self, model_name_or_path: str, device: str, args_dict: Dict[str, Any]):
        super().__init__(model_name_or_path, device, args_dict)

    def _initialize_model(self):
        self.quantization_config = self._get_quantization_config()

        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.bfloat16,
            quantization_config=self.quantization_config,
            device_map=self.device
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_fast = False,
            quantization_config=self.quantization_config
        )
        self.model.eval()

    def _get_quantization_config(self):
        return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )

    def process_image(self, system_prompt: str, user_prompt: str, image: Image.Image) -> str:
        image = resize_image_proportionally(image, 768, 768)

        user_only_prompt = system_prompt + " " + user_prompt

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_only_prompt},
                    {"type": "image"},
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                generate_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.3, use_cache=True, top_k=20)
        caption = self.processor.batch_decode(generate_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        return caption

class Idefics3Handler(ModelHandler):
    def __init__(self, model_name_or_path: str, device: str, args_dict: Dict[str, Any]):
        super().__init__(model_name_or_path, device, args_dict)        

    def _initialize_model(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        self.model.eval()

    def process_image(self, system_prompt: str, user_prompt: str, image: Image.Image) -> str:
        image = resize_image_proportionally(image, 2048, 2048)
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

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            caption = generated_texts[0].split("Assistant: ")[1]
        return caption

class LlavaHandler(ModelHandler):
    def __init__(self, model_name_or_path: str, device: str, args_dict: Dict[str, Any]):
        super().__init__(model_name_or_path, device, args_dict)        

    def _initialize_model(self):
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16, 
            #low_cpu_mem_usage=True,            
            vision_feature_select_strategy="default",
            attn_implementation='flash_attention_2', # sdpa or flash_attention_2 or "eager"
            device_map=self.device
        )
        self.processor = LlavaNextProcessor.from_pretrained(
            self.model_name_or_path,            
            #padding_side="left",
            #vision_feature_select_strategy="default",
            #patch_size=32,
            use_fast=False
        )
        self.processor.patch_size = self.model.config.vision_config.patch_size
        self.processor.vision_feature_select_strategy = self.model.config.vision_feature_select_strategy
        
        self.model.eval()

    def process_image(self, system_prompt: str, user_prompt: str, image: Image.Image) -> str:
        image = resize_image_proportionally(image, 768, 768)
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

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)    
            caption = generated_texts[0].split("[/INST] ")[1]
        return caption

class MiniCPMoHandler(ModelHandler):
    def __init__(self, model_name_or_path: str, device: str, args_dict: Dict[str, Any]):
        super().__init__(model_name_or_path, device, args_dict)

    def _initialize_model(self):
        self.quantization_config = self._get_quantization_config()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_name_or_path,
            attn_implementation='eager', # sdpa or flash_attention_2 or "eager"
            torch_dtype=torch.bfloat16,            
            trust_remote_code=True,            
            local_files_only=True,            
            init_vision=True,
            init_audio=True,
            init_tts=True,            
            quantization_config=self.quantization_config,
            device_map=self.device
        )

        self.model.eval()
        self.model.init_tts()
        self.model.tts.float()

    def process_image(self, system_prompt: str, user_prompt: str, image: Image.Image) -> str:
        image = resize_image_proportionally(image, 2048, 2048)

        user_only_prompt = system_prompt + " " + user_prompt

        msgs = [{'role': 'user', 'content': [image, user_only_prompt]}]
        caption = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.tokenizer
        )

        return caption

class GenericModelHandler(ModelHandler):
    def __init__(self, model_name_or_path: str, device: str, args_dict: Dict[str, Any]):
        super().__init__(model_name_or_path, device, args_dict)

    def _initialize_model(self):
        self.quantization_config = self._get_quantization_config()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,                        
            quantization_config=self.quantization_config,
            device_map=self.device
        )

        self.model.eval()

    def process_image(self, system_prompt: str, user_prompt: str, image: Image.Image) -> str:
        image = resize_image_proportionally(image, 1024, 1024)

        user_only_prompt = system_prompt + " " + user_prompt

        msgs = [{'role': 'user', 'content': [image, user_only_prompt]}]
        caption = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.tokenizer
        )

        return caption    
