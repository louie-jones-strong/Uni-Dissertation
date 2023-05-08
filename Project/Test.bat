call .venv-dev\Scripts\activate.bat
cls
py -m mypy Main.py --config-file config.ini
unittest-parallel -q
flake8 src/ --config=config.ini
flake8 Tests/ --config=config.ini