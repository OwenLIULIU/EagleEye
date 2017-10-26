#!/bin/bash

# create and enter virtualenv
if [ ! -d "env" ]; then
  virtualenv env
fi
. env/bin/activate

# install requirements
pip install -r requirements.txt
