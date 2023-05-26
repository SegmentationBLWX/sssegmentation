#!/bin/bash
THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $THIS_DIR
cd ..

CFGFILEPATH=$1
CKPTSPATH=$2

python ssseg/inference.py \
    --cfgfilepath $CFGFILEPATH \
    --ckptspath $CKPTSPATH ${@:3}