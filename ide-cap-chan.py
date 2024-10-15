import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from tqdm import tqdm
from pathlib import Path
from os.path import join as opj, exists as exts
from os import walk
from gc import collect as gcollect
from argparse import ArgumentParser

parser = ArgumentParser(description='Generate captions for images')
parser.add_argument('--input_dir', type=str, default=".\\2tag", help='Path to the directory containing images')
parser.add_argument('--CUDA_VISIBLE_DEVICES', type=int, default=0, help='Use specified CUDA device')
parser.add_argument('--use_tags', type=bool, default=True, help='Use tags to enhance captioning')
parser.add_argument('--caption_suffix', type=str, default=".txt", help='Suffix for the generated caption files')
parser.add_argument('--tags_suffix', type=str, default=".ttxt", help='Suffix for the tag files')
args = parser.parse_args()

#model_name_or_path="Minthy/ToriiGate-v0.3"
model_name_or_path = ".\\ToriiGate-v0.3"
caption_suffix=args.caption_suffix #suffix for generated captions
tags_suffix=args.tags_suffix #suggix for file with booru tags
use_tags=args.use_tags #set to True for using with reference tags
image_extensions=[".jpg",".png",".webp",".jpeg"]

# quantization_config = BitsAndBytesConfig(
   # load_in_4bit=True,
   # bnb_4bit_quant_type="nf4",
   # bnb_4bit_use_double_quant=True,
   # bnb_4bit_compute_dtype=torch.bfloat16
# )

#quantization_config = BitsAndBytesConfig(load_in_8bit=True)

DEVICE = "cuda:" + str(args.CUDA_VISIBLE_DEVICES)
processor = AutoProcessor.from_pretrained(model_name_or_path) #or change to local path
model = AutoModelForVision2Seq.from_pretrained(
        model_name_or_path, 
        torch_dtype=torch.bfloat16,
        # quantization_config=quantization_config,
        # device_map="auto",
        # _attn_implementation="flash_attention_2", #if installed
).to(DEVICE)

for root, dirs, files in walk(args.input_dir):
    filelist = [file for file in files if any([file.endswith(ext) for ext in image_extensions])]
    for fn in tqdm(filelist, desc="Captioning"):
        caption_name = opj(root, fn.split(".")[0]+caption_suffix)
        print("Caption filename: " + caption_name)
        if exts(caption_name):
            continue
    
        print("Image: " + opj(root,fn))
        image = load_image(opj(root,fn))
          
        ###Trained options
        #user_prompt="Describe the picture in structuted json-like format."
        user_prompt="Give a long and detailed description of the picture."
        #user_prompt="Describe the picture briefly."
        ###Any other questions or instructions
        #user_prompt="What color is the ribbon in the character"s hair?"
        #...
        
        #Optional, add booru tags
        if use_tags:
            try:
                tag_caption_name = opj(root, fn.split(".")[0]+tags_suffix)
                if exts(tag_caption_name):
                    print("Booru tags filename: " + tag_caption_name)
                    tags=open(tag_caption_name).read().strip()
                    user_prompt+=" Also here are booru tags for better understanding of the picture, you can use them as reference."
                    user_prompt+=f" <tags>\n{tags}\n</tags>"
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
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Generate
        model.eval()
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=500)
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            caption=generated_texts[0].split("Assistant: ")[1]
            gcollect()
            torch.cuda.empty_cache()
              
        with open(caption_name,"w",encoding="utf-8",errors="ignore") as outf:                       
            outf.write(caption)