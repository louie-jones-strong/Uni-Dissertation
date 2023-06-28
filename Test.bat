call .venv\Scripts\activate.bat
cls

@REM Type checking
py -m mypy src/ --config-file config.ini

@REM Unit tests
unittest-parallel -p *Test*.py

@REM Linting
py -m flake8 src --config=config.ini
py -m flake8 Tests --config=config.ini