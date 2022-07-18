#!/bin/bash
THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $THIS_DIR
cd ..

NGPUS=$1
CFGFILEPATH=$2
CHECKPOINTSPATH=$3
PORT=${PORT:-6666}
NNODES=${NNODES:-1}
NODERANK=${NODERANK:-0}
MASTERADDR=${MASTERADDR:-"127.0.0.1"}

python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODERANK \
    --master_addr=$MASTERADDR \
    --nproc_per_node=$NGPUS \
    --master_port=$PORT \
    ssseg/test.py --nproc_per_node $NGPUS \
                  --cfgfilepath $CFGFILEPATH \
                  --checkpointspath $CHECKPOINTSPATH ${@:4}