@echo off
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" face_enhancement.py ^
--model GPEN-BFR-512 ^
--in_size 512 ^
--use_sr

pause