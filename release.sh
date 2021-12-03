#! /bin/bash
docformatter --in-place **/*.py --wrap-summaries 88 --wrap-descriptions 88
isort --atomic **/*.py
black .

coverage run --source fse setup.py test