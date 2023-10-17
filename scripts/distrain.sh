#!/bin/bash
THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $THIS_DIR
cd ..

NGPUS=$1
CFGFILEPATH=$2
PORT=${PORT:-$(($RANDOM+8888))}
NNODES=${NNODES:-1}
NODERANK=${NODERANK:-0}
MASTERADDR=${MASTERADDR:-"127.0.0.1"}
TORCHVERSION=`python -c 'import torch; print(torch.__version__)'`

if [[ $TORCHVERSION == "2."* ]]; then
    torchrun ssseg/train.py --nproc_per_node $NGPUS --cfgfilepath $CFGFILEPATH ${@:3}
else
    python -m torch.distributed.launch \
        --nnodes=$NNODES \
        --node_rank=$NODERANK \
        --master_addr=$MASTERADDR \
        --nproc_per_node=$NGPUS \
        --master_port=$PORT \
        ssseg/train.py --nproc_per_node $NGPUS \
                    --cfgfilepath $CFGFILEPATH ${@:3}
fi