call .\venv\Scripts\activate.bat
python "ide-cap-chan.py"

rem Full command line args example:
rem python "ide-cap-chan3.py" --model_path "model_name" --model_type "model_type" --input_dir "folder_with_images_to_tag" --CUDA_VISIBLE_DEVICES "0, 1" --caption_suffix ".txt" --dont_use_tags --tags_suffix ".ttxt"

rem Local models
rem python "ide-cap-chan.py" --model_path "Minthy_ToriiGate-v0.4-2B-exl2-8bpw"
rem python "ide-cap-chan.py" --model_type "idefics3" --model_path "ToriiGate-v0.3" --CUDA_VISIBLE_DEVICES "1"
rem python "ide-cap-chan.py" --model_type "idefics3" --model_path "ToriiGate-v0.3-nf4" --CUDA_VISIBLE_DEVICES "0"
rem python "ide-cap-chan.py" --model_type "idefics3" --model_path "ToriiGate-v0.3" --CUDA_VISIBLE_DEVICES "0" --add_tags --caption_suffix ".text" --tags_suffix ".txt"
rem python "ide-cap-chan.py" --model_type "qwen2vl" --model_path "Minthy_ToriiGate-v0.4-2B" --CUDA_VISIBLE_DEVICES "0" 
rem python "ide-cap-chan.py" --model_type "qwen2vl" --model_path "Vikhr-2-VL-2b-Instruct-experimental" --CUDA_VISIBLE_DEVICES "0"
rem python "ide-cap-chan.py" --model_type "minicpmo" --model_path "MiniCPM-o-2_6" --CUDA_VISIBLE_DEVICES "0"
rem python "ide-cap-chan.py" --model_type "qwen2vl" --model_path "Minthy_ToriiGate-v0.4-7B" --CUDA_VISIBLE_DEVICES "0" --no_chars --add_tags --caption_format "markdown" --caption_suffix ".text" --tags_suffix ".txt"
rem python "ide-cap-chan.py" --model_type "exllama2" --model_path "Minthy_ToriiGate-v0.4-2B-exl2-8bpw" --CUDA_VISIBLE_DEVICES "0" --no_chars --add_tags --caption_format "bbox" --caption_suffix ".text" --tags_suffix ".txt"
rem python "ide-cap-chan.py" --model_type "exllama2" --model_path "Minthy_ToriiGate-v0.4-2B-exl2-8bpw" --CUDA_VISIBLE_DEVICES "0, 1"
rem python "ide-cap-chan.py" --model_type "exllama2" --model_path "Minthy_ToriiGate-v0.4-7B-exl2-8bpw" --CUDA_VISIBLE_DEVICES "0"
rem python "ide-cap-chan.py" --model_type "pixtral" --model_path "Ertugrul_Pixtral-12B-Captioner-Relaxed" --CUDA_VISIBLE_DEVICES "0"
rem python "ide-cap-chan.py" --model_type "molmo" --model_path "ctranslate2-4you_molmo-7B-O-bnb-4bit" --CUDA_VISIBLE_DEVICES "0"
rem python "ide-cap-chan.py" --model_type "molmo72b" --model_path "SeanScripts_Molmo-72B-0924-nf4"
rem python "ide-cap-chan.py" --model_type "qwen2vl" --model_path "Ertugrul_Qwen2-VL-7B-Captioner-Relaxed" --CUDA_VISIBLE_DEVICES "0"
rem python "ide-cap-chan.py" --model_type "idefics3" --model_path "Idefics3-8B-Llama3"
rem python "ide-cap-chan.py" --model_type "llava" --model_path "llava-v1.6-mistral-7b-hf-nf4" --CUDA_VISIBLE_DEVICES "0"
rem python "ide-cap-chan.py" --model_type "llava" --model_path "llava-v1.6-mistral-7b-hf" --CUDA_VISIBLE_DEVICES "0"
rem python "ide-cap-chan.py" --model_type "joy-caption" --model_path "llama-joycaption-alpha-two-hf-llava" --CUDA_VISIBLE_DEVICES "0"
rem python "ide-cap-chan.py" --model_type "llava" --model_path "llama-joycaption-alpha-two-hf-llava" --CUDA_VISIBLE_DEVICES "0"