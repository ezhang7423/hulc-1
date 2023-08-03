#!/bin/bash

cd calvin_env_repo
./install.sh

pip install -e ./hiveformer

cd ..
pip install -e .
