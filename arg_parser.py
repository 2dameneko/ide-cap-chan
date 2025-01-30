from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser(description='Generate captions for images')
    parser.add_argument('--model_path', type=str, default="", help='Path to the used model')
    parser.add_argument('--model_type', type=str, default="exllama2", help='Model type (supported architectures: idefics3, llava, joy-caption, molmo, qwen2vl, molmo72b, pixtral, exllama2)')
    parser.add_argument('--input_dir', type=str, default="./2tag", help='Path to the folder containing images')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default="0", help='Comma-separated list of CUDA devices. WARNING: multi-GPU captioning can overload your power supply unit. Model molmo72b ignore this arg and require 2x24GB GPU')
    parser.add_argument('--caption_suffix', type=str, default=".txt", help='Extension for generated caption files')
    parser.add_argument('--dont_use_tags', default=False, action='store_true', help='Do not use existing *booru tags to enhance captioning')
    parser.add_argument('--tags_suffix', type=str, default=".ttxt", help='Extension for existing *booru tag files')
    return parser.parse_args()
