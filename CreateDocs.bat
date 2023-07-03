call .venv\Scripts\activate.bat

sphinx-apidoc -o docs .
call docs\make html