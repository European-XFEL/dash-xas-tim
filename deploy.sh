#!/usr/bin/env bash

ENV_DIR=${PWD}/env
conda create -y -p ${ENV_DIR} python=3.7
${ENV_DIR}/bin/python -m pip install -I -r requirements.txt

echo "${ENV_DIR}/bin/python ${PWD}/app.py" > dash-xas-tim