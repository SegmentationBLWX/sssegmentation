#!/bin/bash
THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $THIS_DIR
cd ..
DATASETNAME=$1
BACKBONENAME=$2
MODELNAME=$3
CHECKPOINTSPATH=$4
python3 ssseg/train.py --datasetname $DATASETNAME --backbonename $BACKBONENAME --modelname $MODELNAME ${4}