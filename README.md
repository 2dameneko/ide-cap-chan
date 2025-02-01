# ide-cap-chan

<div align="center">
    <img src="https://count.getloli.com/get/@ide-cap-chan?theme=asoul&padding=4" alt="Visitor count"><br>
</div>

ide-cap-chan is a utility for batch captioning images with natural language using various Vision-Language (VL) models.

## Features
* **High-speed processing**: Optimized for rapid batch caption generation with Qwen2-VL-7B-Instruct, Idefics3-8B-Llama3, llava-v1.6, Llama JoyCaption Alpha Two, Molmo-7B-D, Molmo-7B-O, Molmo-72B, and Pixtral models
* **Multi-GPU support**: Distribute workloads across multiple GPUs
* **Efficient quantization**: Supports ExLlama2 (exl2), fp16, and nf4 quantization for reduced VRAM usage
* **Autoload strategies**: VRAM-optimized loading for all exl2 and fp16 Molmo-72B models
* **Model flexibility**: Use default or custom models via CLI arguments.
* **Input flexibility**: Supports Hugging Face, local, and external models
* **Tag integration**: Enhance captions with existing tags/captions
* **Process control**: Interrupt and resume captioning tasks
* **Batch processing**: Recursively process subfolders in input directories

## Requirements
* NVIDIA GPU with CUDA support (8GB VRAM minimum for llava, 12GB recommended for Qwen2-VL-7B in exl2, 2x24GB for Molmo-72B)

## Installation
1. Clone the repository:  
   `git clone https://github.com/2dameneko/ide-cap-chan`
2. Install dependencies:
   - **Windows**: Run `install.bat`
   - **Linux**: Create a virtual environment and install requirements:  
     ```bash
     python -m venv venv
     source venv/bin/activate
     pip install -r requirements.txt
     ```

## Usage
1. Place images and corresponding tag files in the input folder (default: `2tag`)
2. Start processing:
   - **Windows**: Run `batch_processing.bat`
   - **Linux**: Execute `python ide-cap-chan.py`
3. Specify alternative models using CLI arguments
4. Customize prompts in `model_handler.py` (modify `system_prompt` and `user_prompt`)

## Updating
- **Windows**: Run `update.cmd`

## Options
Run without arguments for default behavior. Available CLI options (`python ide-cap-chan.py -h`):
| Argument | Description |
|----------|-------------|
| `--model_path` | Path to model (Hugging Face, local, or external) |
| `--model_type` | Model architecture/loader: `idefics3`, `llava`, `joy-caption`, `molmo`, `molmo72b`, `qwen2vl`, `pixtral`, `exllama2` (default: `exllama2`) |
| `--input_dir` | Input directory path (default: `2tag`) |
| `--CUDA_VISIBLE_DEVICES` | Comma-separated GPU IDs (default: `0`). **Note**:<br>- Multi-GPU may strain power supplies<br>- `molmo72b` ignores this argument and auto-splits across GPUs |
| `--caption_suffix` | Caption file extension (default: `.ttxt`) |
| `--tags_suffix` | Tag file extension (default: `.txt`) |
| `--caption_format` | Output format: `json`, `markdown`, `short`, `long`, `bbox` (requires ToriiGate ≥0.4) |
| `--add_tags` | Enhance captions with existing tag files (ToriiGate-family models) |
| `--add_chars` | Include character information (requires ToriiGate ≥0.4) |
| `--add_char_traits` | Include character traits (requires ToriiGate ≥0.4) |
| `--add_info` | Include miscellaneous image info (requires ToriiGate ≥0.4) |
| `--no_chars` | Do not add character information (requires ToriiGate ≥0.4) |

## Supported File Formats
`.jpg`, `.png`, `.webp`, `.jpeg`

## Version History
* **0.8**: Added ExLlama2 loader support (default), ToriiGate-v0.4 features, Molmo-72B auto-split
* **0.7**: Added Molmo/Qwen2VL/Pixtral support, improved multi-GPU quant processing, code refactor
* **0.6**: Internal code improvements
* **0.5**: Added JoyCaption support, code refactor
* **0.4**: Added LLaVA support, updated to PyTorch 2.5.1
* **0.3**: Improved argument handling, fixed extension case sensitivity
* **0.2**:  
  - Multi-GPU support with load balancing  
  - nf4 quantization
  - Fixed duplicate file filtering  
  - Updated environment scripts  
* **0.1**: Initial release

## Note
This project is a proof of concept and not production-ready.

## License
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

## Credits
- Idefics3 Architecture: [HuggingFaceM4/Idefics3-8B-Llama3](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3)
- LLaVA Architecture: [Transformers Documentation](https://huggingface.co/docs/transformers/main/model_doc/llava)
- JoyCaption Code: [fpgaminer/joycaption](https://github.com/fpgaminer/joycaption)
- Qwen2-VL Architecture: [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- Qwen2-VL Implementation: [MNeMoNiCuZ/qwen2-vl-7b-captioner-relaxed-batch](https://github.com/MNeMoNiCuZ/qwen2-vl-7b-captioner-relaxed-batch)
- Molmo Architecture: [AllenAI Collection](https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19)
- Pixtral Architecture: [Pixtral Documentation](https://huggingface.co/docs/transformers/model_doc/pixtral)

**Model Credits**  
[ToriiGate](https://huggingface.co/Minthy) · [LLaVA](https://huggingface.co/llava-hf) · [JoyCaption](https://huggingface.co/fancyfeast) · [Qwen2, Pixtral](https://huggingface.co/Ertugrul) · [Molmo](https://huggingface.co/cyan2k) · [Molmo72b](https://huggingface.co/SeanScripts/Molmo-72B-0924-nf4)

Thank you for your interest in ide-cap-chan!