call .venv\Scripts\activate.bat
cd docs

sphinx-apidoc -o . ..
make html
cd ../