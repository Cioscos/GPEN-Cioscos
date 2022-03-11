@echo off
call _internal\setenv.bat

"%PYTHON_EXECUTABLE%" segmentation2face.py

pause