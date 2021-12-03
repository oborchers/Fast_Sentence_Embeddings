#! /bin/bash

coverage run --source fse setup.py test
coverage-badge -o coverage.svg
mv coverage.svg media/