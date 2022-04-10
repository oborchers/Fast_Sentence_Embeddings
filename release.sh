#! /bin/bash
docformatter --in-place **/*.py --wrap-summaries 88 --wrap-descriptions 88
isort --atomic **/*.py
black .

pytest -v --cov=fse --cov-report=term-missing

handsdown