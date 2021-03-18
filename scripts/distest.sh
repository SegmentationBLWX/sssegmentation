#!/bin/bash
THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $THIS_DIR
cd ..
NGPUS=$1
CFGFILEPATH=$2
CHECKPOINTSPATH=$3
EVALMODE=${EVALMODE:-"offline"}
PORT=${PORT:-6666}
python3 -m torch.distributed.launch --nproc_per_node $NGPUS --master_port $PORT ssseg/test.py --nproc_per_node $NGPUS --cfgfilepath $CFGFILEPATH --checkpointspath $CHECKPOINTSPATH --evalmode $EVALMODE