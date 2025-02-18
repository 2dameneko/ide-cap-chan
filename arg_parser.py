from argparse import ArgumentParser
import sys

def parse_arguments():
    parser = ArgumentParser(description='Generate captions for images')

    parser.add_argument('--model_path', type=str, default="", help='Path to the used model')
    parser.add_argument('--model_type', type=str, default="exllama2",
                        help='Model type (supported architectures: idefics3, llava, joy-caption, molmo, qwen2vl, molmo72b, pixtral, exllama2, minicpmo, generic (You can try this option if your model is not listed as supported. No warranties.))')
    parser.add_argument('--input_dir', type=str, default="./2tag", help='Path to the folder containing images')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default="0",
                        help='Comma-separated list of CUDA devices. WARNING: multi-GPU captioning can overload your power supply unit. Model molmo72b ignores this arg and requires 2x24GB GPU')
    parser.add_argument('--caption_suffix', type=str, default=".txt", help='File extension for generated caption files')
    parser.add_argument('--tags_suffix', type=str, default=".ttxt", help='File extension for existing image info file (like *booru tags, traits, characters, etc)')
    parser.add_argument('--caption_format', type=str, choices=['json', 'markdown', 'short', 'long', 'bbox'], default='long',
                        help='Format of the generated captions (supported formats: json, markdown, short, long, bbox), (req. ToriiGate-family models)')

    bool_args = [
        ('--add_tags', 'Use an additional file as existing *booru tags to enhance captioning, (req. ToriiGate-family models)'),
        ('--add_chars', 'Use an additional file as information about represented characters, (req. ToriiGate >= 0.4 model)'),
        ('--add_char_traits', 'Use an additional file as information about character traits, (req. ToriiGate >= 0.4 model)'),
        ('--add_info', 'Use an additional file as misc information about image, (req. ToriiGate >= 0.4 model)'),
        ('--no_chars', 'Do not add any characters to the output, (req. ToriiGate >= 0.4 model)'),
    ]

    for arg, help_text in bool_args:
        parser.add_argument(arg, default=False, action='store_true', help=help_text)

    args = parser.parse_args()

    check_mutually_exclusive(
        args, ['--add_tags', '--add_chars', '--add_char_traits', '--add_info']
    )

    check_mutually_exclusive(
        args, ['--add_chars', '--no_chars']
    )

    return args

def check_mutually_exclusive(args, arg_names):
    args_list = [getattr(args, arg_name.replace('--', '')) for arg_name in arg_names]
    if sum(args_list) > 1:
        print(f"Error: Only one of the following arguments can be True at a time: {', '.join(arg_names)}")
        sys.exit(1)
