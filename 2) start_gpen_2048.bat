@echo off
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" face_enhancement.py ^
--model GPEN-BFR-2048 ^
--in_size 2048 ^
--use_cuda

pause