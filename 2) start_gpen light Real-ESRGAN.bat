@echo off
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" face_enhancement.py ^
--model GPEN-BFR-256 ^
--use_sr ^
--in_size 256 ^
--narrow 0.5 ^
--channel_multiplier 1 ^
--use_cuda

pause