py -3.8 -m pip install virtualenv
py -3.8 -m virtualenv .venv
call .venv\Scripts\activate.bat
py -m pip install --upgrade pip


pip install -r requirements-dev.txt

pip install -r src\ExperienceStore\requirements.txt
pip install -r src\Learner\requirements.txt
pip install -r src\WebServer\requirements.txt
pip install -r src\Worker\requirements.txt