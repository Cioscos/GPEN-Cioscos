@echo off
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" train_simple.py --size 256 --channel_multiplier 1 --narrow 0.5 --ckpt .\training_weight\010000.pth --sample .\results\sample --batch 1 --path input\training-data

pause