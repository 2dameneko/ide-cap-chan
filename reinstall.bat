python -m venv venv
call venv\Scripts\activate
rem pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pause
