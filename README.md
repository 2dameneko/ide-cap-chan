# ide-cap-chan
ide-cap-chan is a utility for batch captioning images with natural language using models with Idefics3-8B-Llama3 or llava-v1.6 architecture.

## Features
* Batch caption generation for Idefics3-8B-Llama3, llava-v1.6, Llama JoyCaption Alpha Two, Qwen2-VL-7B-Instruct, Molmo-7B-D, Molmo-7B-O, Molmo-72B, Pixtral models
* Support for multi-GPU captioning
* Support of nf4 quants for lower VRAM requirements:
  - [ToriiGate-v0.3-nf4](https://huggingface.co/2dameneko/ToriiGate-v0.3-nf4)
  - [Idefics3-8B-Llama3-nf4](https://huggingface.co/2dameneko/Idefics3-8B-Llama3-nf4)
  - [llava-v1.6-mistral-7b-hf-nf4](https://huggingface.co/2dameneko/llava-v1.6-mistral-7b-hf-nf4)
  - [Llama JoyCaption Alpha Two](https://huggingface.co/fancyfeast/llama-joycaption-alpha-two-hf-llava)  
  - [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
  - [Molmo](https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19)
  - [Pixtral](https://huggingface.co/docs/transformers/model_doc/pixtral)
  
* Support for huggingface/local/external models
* Support for additional tag files to enhance captions
* Interrupting and resuming the captioning process
* Recursive processing of subfolders in the specified input folder

## Requirements
* A video card with CUDA support (from 8GB for llava up to 24GB Qwen2/Molmo7B and 2x24GB for Molmo 72B)

## Installation
1. Clone the repository: `git clone https://github.com/2dameneko/ide-cap-chan`
2. On Windows: run `install.bat`, on Linux: make venv and run `pip install -r requirements.txt`

## Usage
1. Place images and corresponding tag files in the input folder (default: `2tag`).
2. On Windows: run `batch_processing.bat`, on Linux: run the script with the following command: `python ide-cap-chan.py`
3. You can use different models of supported architectures by specifying them on the command line
4. You can optionally modify the prompt according to your specific conditions in the model_handler.py file: system_prompt and user_prompt.
    * Note: `molmo72b` use it's own shorter promt to prevent OOM

## Update
1. On Windows: run `update.cmd`

## Options
By default, no command line arguments are required.
Additional command line arguments: `python ide-cap-chan.py -h`
* `--model_path` - Path to the used model. Default `cyan2k/molmo-7B-O-bnb-4bit`
* `--model_type` - Model type (supported architectures: idefics3, llava, joy-caption, molmo, molmo72b, qwen2vl, pixtral). Default `molmo`
* `--input_dir` - Path to the folder containing images. Default `2tag`
* `--CUDA_VISIBLE_DEVICES` comma-separated list of CUDA devices. Default `0`. 
    * WARNING: multi-GPU captioning can overload your power supply unit
    * Note: `molmo72b` model ignore CUDA_VISIBLE_DEVICES arg and use 0 and 1 GPUs
* `--caption_suffix` Extension for generated caption files. Default `.ttxt`
* `--dont_use_tags` Don't use existing *booru tags to enhance captioning. Default `False`
* `--tags_suffix` Extension for existing *booru tag files. Default `.txt`'Extension for existing *booru tag files'

## File formats supported:
`.jpg`, `.png`, `.webp`,`.jpeg`

## Version history
* 0.7: Added `molmo`, `molmo72b`, `qwen2vl`, `pixtral` architectures support. Set default to `molmo`. Fixed quants milti-GPU processing - at full speed now. Project structure changed. Refactored.
* 0.6: Refactored, internal.
* 0.5: Added `joy-caption` architecture support. Refactored.
* 0.4: Added `llava` architecture support. Reworked args. Removed temporally pinned pytorch ver to 2.4.1 due bugged 2.5 release. Now it's all ok with pytorch 2.5.1
* 0.3: Reworked 'using' args, fixed minor bug with file extension case
* 0.2:
  * Support for multi-GPU captioning (-h for command line args) with proportional workload balancing
  * Support of nf4 quants, enabled by default. ~5Gb model size instead of ~18Gb
  * Fixed filtering bug with same name files for captioning in different folders in one batch
  * Reworked scripts for VENV creation and update
  * Code refactoring
* 0.1 Inital release

## License
[https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)

## Credits
Idefics3 Finetuned model: [https://huggingface.co/Minthy/ToriiGate-v0.3](https://huggingface.co/Minthy/ToriiGate-v0.3)
Ifdefics3 Architecture and model: [https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3)
llava Architecture: [https://huggingface.co/docs/transformers/main/model_doc/llava](https://huggingface.co/docs/transformers/main/model_doc/llava)
llava model: [https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)
Portions of code for joy-caption support: [https://github.com/fpgaminer/joycaption](https://github.com/fpgaminer/joycaption)
Joy-caption model: [https://huggingface.co/fancyfeast/llama-joycaption-alpha-two-hf-llava](https://huggingface.co/fancyfeast/llama-joycaption-alpha-two-hf-llava)
Qwen2-VL-7B-Instruct Architecture: [https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
Qwen2-VL-7B-Instruct Finetuned Model: [https://huggingface.co/Ertugrul/Qwen2-VL-7B-Captioner-Relaxed](https://huggingface.co/Ertugrul/Qwen2-VL-7B-Captioner-Relaxed)
Portions of code for Qwen2-VL-7B support: [https://github.com/MNeMoNiCuZ/qwen2-vl-7b-captioner-relaxed-batch](https://github.com/MNeMoNiCuZ/qwen2-vl-7b-captioner-relaxed-batch)
Molmo Architecture: [https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19](https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19)
Molmo Quantized 7B model: [https://huggingface.co/cyan2k/molmo-7B-O-bnb-4bit](https://huggingface.co/cyan2k/molmo-7B-O-bnb-4bit)
Molmo Quantized 72B model: [https://huggingface.co/SeanScripts/Molmo-72B-0924-nf4](https://huggingface.co/SeanScripts/Molmo-72B-0924-nf4)
Pixtral Architecture: [https://huggingface.co/docs/transformers/model_doc/pixtral](https://huggingface.co/docs/transformers/model_doc/pixtral)
Pixtral Finetuned Model: [https://huggingface.co/Ertugrul/Pixtral-12B-Captioner-Relaxed](https://huggingface.co/Ertugrul/Pixtral-12B-Captioner-Relaxed)

Thank you for your interest in ide-cap-chan!