py -3.10 -m pip install virtualenv
py -3.10 -m virtualenv .venv
call .venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
pip install -e ../../../baselines
