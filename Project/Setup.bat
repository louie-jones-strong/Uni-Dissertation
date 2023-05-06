py -3.10 -m pip install virtualenv
py -3.10 -m virtualenv .venv
call .venv\src\activate.bat
py -m pip install --upgrade pip
pip install -r requirements.txt
py src/Main.py