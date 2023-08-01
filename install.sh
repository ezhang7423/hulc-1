#!/bin/bash

cd calvin
./install.sh

pip install -e ./hiveformer

cd ..
pip install -e .