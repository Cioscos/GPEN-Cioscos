@echo off
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" face_inpainting.py

pause