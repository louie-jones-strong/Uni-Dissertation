py -3.10 -m pip install virtualenv
py -3.10 -m virtualenv .venv-dev
call .venv-dev\Scripts\activate.bat
py -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements_dev.txt