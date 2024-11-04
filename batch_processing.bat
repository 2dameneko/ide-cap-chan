call .\venv\Scripts\activate.bat
python "ide-cap-chan.py"

rem Full comand line args example:
rem python "ide-cap-chan3.py" --model_path "model_name" --model_type "model_type" --input_dir "folder_twith_images_to_tag" --CUDA_VISIBLE_DEVICES "0, 1" --caption_suffix ".txt" --dont_use_tags --tags_suffix ".ttxt"

rem Use llava model
rem python "ide-cap-chan.py" --model_type "llava"

rem Local models
rem python "ide-cap-chan3.py" --model_path "Idefics3-8B-Llama3"
rem python "ide-cap-chan.py" --model_path "ToriiGate-v0.3"
rem python "ide-cap-chan3.py" --model_path "llava-v1.6-mistral-7b-hf" --model_type "llava"