@echo off
set PYTHON_PATH=%~dp0ProjectInterpreter\Scripts
"%PYTHON_PATH%\pyinstaller"  .\snakeml_onefile_build.spec