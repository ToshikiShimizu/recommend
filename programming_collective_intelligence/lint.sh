autopep8 -i */*.py
pytest -s --cov=src
python3 src/main.py
