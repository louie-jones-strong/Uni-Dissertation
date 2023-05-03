call .venv-dev\Scripts\activate.bat
cls
py -m mypy Scripts/Main.py --config-file Tests/mypy.ini --install-types