call .venv\Scripts\activate.bat
cls
py -m mypy src/ExperienceStore.py --config-file config.ini
py -m mypy src/Learner.py --config-file config.ini
py -m mypy src/WebServer.py --config-file config.ini
py -m mypy src/Worker.py --config-file config.ini
unittest-parallel -q
flake8 src/ --config=config.ini
flake8 Tests/ --config=config.ini