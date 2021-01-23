#!/bin/bash
THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $THIS_DIR
cd ..
CFGFILEPATH=$1
CHECKPOINTSPATH=${2:-"latest.pth"}
python3 ssseg/train.py --cfgfilepath $CFGFILEPATH --checkpointspath $CHECKPOINTSPATH