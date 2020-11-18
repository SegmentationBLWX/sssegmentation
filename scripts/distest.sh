#!/bin/bash
THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $THIS_DIR
cd ..
NGPUS=$1
CFGFILEPATH=$2
CHECKPOINTSPATH=$3
NOEVAL=$4
python3 -m torch.distributed.launch --nproc_per_node $NGPUS ssseg/test.py --nproc_per_node $NGPUS --cfgfilepath $CFGFILEPATH --checkpointspath $CHECKPOINTSPATH ${4}