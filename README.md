# ide-cap-chan
ide-cap-chan is a utility for batch captioning images with natural language using models with Idefics3-8B-Llama3 architecture.

## Features
* Batch caption generation for Idefics3-8B-Llama3 models.
* Support for additional tag files to enhance captions.
* Interrupting and resuming the captioning process.
* Recursive processing of subfolders in the specified input folder.

## Requirements
* A video card with 24GB VRAM is required.

## Installation
1. Clone the repository: `git clone https://github.com/your-username/ide-cap-chan`
2. On Windows: run `install.bat`, on Linux = make venv and run `pip install -r requirements.txt`

## Usage
1. Place images and corresponding tag files in the input folder (default: `2tag`).
2. On Windows: run `batch_processing.bat`, on Linux: run the script with the following command: `python ide-cap-chan.py`

## Options
Additional command line arguments: `python ide-cap-chan.py -h`

## File formats supported:
`.jpg`, `.png`, `.webp`,`.jpeg`

## License
[https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)


## Thanks
Architecture: [https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3)
Finetune: [https://huggingface.co/Minthy/ToriiGate-v0.3](https://huggingface.co/Minthy/ToriiGate-v0.3)

Thank you for your interest in ide-cap-chan!
