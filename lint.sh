autopep8 -i */*.py
pytest -s --cov=src --cov-report=term-missing
python3 src/main.py
