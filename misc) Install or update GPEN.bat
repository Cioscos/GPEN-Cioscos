@echo off
call _internal\setenv.bat

if exist ".\.git\" (
    goto :pull
)

@echo on

rmdir .\DFLIMG
rmdir .\face_detect
rmdir .\face_model
rmdir .\face_parse
rmdir .\sr_model
del /f __init_paths.py
del /f align_faces.py
del /f face_colorization.py
del /f face_enhancement.py
del /f face_inpainting.py
del /f segmentation2face.py

@echo off
git clone https://github.com/Cioscos/GPEN-Cioscos.git
robocopy /e .\GPEN-Cioscos . /MOVE

:pull
git pull

pause
