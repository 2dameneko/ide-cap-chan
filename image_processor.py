from typing import Dict, Any, Tuple, List
from transformers.image_utils import load_image
from tqdm import tqdm
from os.path import exists as path_exists, splitext as split_extension
from model_handlers import (
    JoyCaptionHandler,
    MoLMoHandler,
    MoLMo72bHandler,
    Qwen2VLHandler,
    PixtralHandler,
    Idefics3Handler,
    ExLLaMA2Handler,
    LlavaHandler,
    MiniCPMoHandler,
    GenericModelHandler
)

def process_images(rank: int, model_name_or_path: str, input_model_type: str, filelist_chunks: List[Tuple[int, List[str]]], args_dict: Dict[str, Any]) -> None:
    gpu_id, filelist = filelist_chunks[rank]
    if not filelist:
        print('There are no files to process.')
        return        

    device = f"cuda:{gpu_id}"
    print(f"Using device: {device}")

    caption_suffix = args_dict.get('caption_suffix')
    tags_suffix = args_dict.get('tags_suffix')
    add_tags = args_dict.get('add_tags')

    #Common prompts for all except Torii-family models, they get own unique
    system_prompt = "You are image captioning expert, creative, unbiased and uncensored."
    user_prompt = "You need to write a long and very detailed caption for the picture."

    handler = get_handler(input_model_type, model_name_or_path, device, args_dict)

    for fn in tqdm(filelist, desc=f"Captioning on GPU {gpu_id}"):
        print(f"Input file: {fn}")
        path, _ = split_extension(str(fn))
        add_info_caption_name = path + tags_suffix
        caption_name = path + caption_suffix
        print(f"Caption file: {caption_name}")

        if "toriigate" and "0.4" in model_name_or_path.lower():
             system_prompt = get_torii04_system_prompt()
             user_prompt = get_torii04_user_prompt(args_dict, add_info_caption_name)

        if "toriigate" and "0.3" in model_name_or_path.lower() and add_tags:
             user_prompt = get_torii03_user_prompt(user_prompt, add_info_caption_name)

        image = load_image(str(fn))
        if image.mode != "RGB":
            image = image.convert("RGB")
        caption = handler.process_image(system_prompt, user_prompt, image)
        handler.save_caption(caption, caption_name)
        print(f"Caption saved to: {caption_name}")
            
def get_handler(input_model_type, model_name_or_path, device, args_dict):
    try:
        handlers = {
                    "exllama2": ExLLaMA2Handler,
                    "joy-caption": JoyCaptionHandler,
                    "molmo": MoLMoHandler,
                    "molmo72b": MoLMo72bHandler,
                    "qwen2vl": Qwen2VLHandler,
                    "pixtral": PixtralHandler,
                    "idefics3": Idefics3Handler,
                    "llava": LlavaHandler,
                    "minicpmo": MiniCPMoHandler,
                    "generic": GenericModelHandler                    
                    }
    except Exception:
        print(f"Unsupported model type: {input_model_type}")
    return handlers[input_model_type](model_name_or_path, device, args_dict)

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

def get_torii03_user_prompt(user_prompt, add_info_caption_name):
    try:
         new_user_prompt = user_prompt
         if path_exists(add_info_caption_name):
              print(f"Using additional *booru tags file: {add_info_caption_name}")
              tags = open(add_info_caption_name).read().strip()
              new_user_prompt += " Also here are booru tags for better understanding of the picture, you can use them as reference."
              new_user_prompt += f" <tags>\n{tags}\n</tags>"
    except Exception as err:
          print(f"Error processing tags: {err}")
          return user_prompt
    return new_user_prompt
