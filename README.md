# ide-cap-chan
ide-cap-chan is a utility for batch captioning images with natural language using models with Idefics3-8B-Llama3 or llava-v1.6 architecture.

## Features
* Batch caption generation for Idefics3-8B-Llama3 or llava-v1.6 models
* Support for multi-GPU captioning
* Support of nf4 quants for lower VRAM requirements:
  - [ToriiGate-v0.3-nf4](https://huggingface.co/2dameneko/ToriiGate-v0.3-nf4)
  - [llava-v1.6-mistral-7b-hf-nf4](https://huggingface.co/2dameneko/llava-v1.6-mistral-7b-hf-nf4)
  - [Idefics3-8B-Llama3-nf4](https://huggingface.co/2dameneko/Idefics3-8B-Llama3-nf4)

* Support for huggingface/local/external models
* Support for additional tag files to enhance captions
* Interrupting and resuming the captioning process
* Recursive processing of subfolders in the specified input folder

## Requirements
* A video card with CUDA support

## Installation
1. Clone the repository: `git clone https://github.com/2dameneko/ide-cap-chan`
2. On Windows: run `install.bat`, on Linux: make venv and run `pip install -r requirements.txt`

## Usage
1. Place images and corresponding tag files in the input folder (default: `2tag`).
2. On Windows: run `batch_processing.bat`, on Linux: run the script with the following command: `python ide-cap-chan.py`

## Update
1. On Windows: run `update.cmd`

## Options
By default, no command line arguments are required.
Additional command line arguments: `python ide-cap-chan.py -h`
* `--model_path` - Path to the used model. Default `2dameneko/ToriiGate-v0.3-nf4`
* `--model_type` - Model type (supported arhictectures: Idefics3, llava). Default `idefics3`
* `--input_dir` - Path to the folder containing images. Default `2tag`
* `--CUDA_VISIBLE_DEVICES` comma-separated list of CUDA devices. Default `0`. WARNING: multi-GPU captioning can overload your power supply unit
* `--caption_suffix` Extension for generated caption files. Default `.ttxt`
* `--dont_use_tags` Don't use existing *booru tags to enhance captioning. Default `False`
* `--tags_suffix` Extension for existing *booru tag files. Default `.txt`'Extension for existing *booru tag files')

## File formats supported:
`.jpg`, `.png`, `.webp`,`.jpeg`

## Version history
* 0.4: Added `llava` architecture support. Reworked args. Removed temporally pinned pytorch ver to 2.4.1 due bugged 2.5 release. Now it's all ok with pytorch 2.5.1
* 0.3: Reworked 'using' args, fixed minor bug with file extension case
* 0.2:
  * Support for multi-GPU captioning (-h for command line args) with proportional workload balancing
  * Support of nf4 quants, enabled by default. ~5Gb model size instead of ~18Gb (but only for single GPU)
  * Fixed filtering bug with same name files for captioning in different folders in one batch
  * Reworked scripts for VENV creation and update
  * Code refactoring
* 0.1 Inital release

## License
[https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)

## Credits
Finetune: [https://huggingface.co/Minthy/ToriiGate-v0.3](https://huggingface.co/Minthy/ToriiGate-v0.3)
Architecture: [https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3)

Thank you for your interest in ide-cap-chan!