@echo off
cd /D %~dp0_internal
call setenv.bat
cd Python37
python -m pip install torch==1.9.0+cu111 ^
torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html ^
opencv-python ^
matplotlib ^
scipy ^
pillow ^
requests ^
tqdm

pause