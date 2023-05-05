call .venv-dev\Scripts\activate.bat
cls
py -m mypy Scripts/Main.py --config-file config.ini
unittest-parallel -q
flake8 Scripts/ --config=config.ini