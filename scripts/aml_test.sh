#!/bin/bash
THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $THIS_DIR
cd ..

# accept arguments
if [ "$RANK" -eq 0 ]; then
    MASTER_IP=$(hostname -I)
else
    MASTER_IP=$(ssh node-0 "hostname -I")
fi
CFGFILEPATH=$1
CKPTSPATH=$2
TORCHVERSION=`python -c 'import torch; print(torch.__version__)'`

# echo info
echo "Node Rank: $RANK"
echo "Master IP: $MASTER_IP"
echo "Node Count: $NODE_COUNT"
echo "Master Port: $MASTER_PORT"

# start testing
if [[ $TORCHVERSION == "2."* ]]; then
    torchrun --nnodes=${NODE_COUNT} --nproc_per_node=${AZUREML_NUM_GPUS} --master_addr=${MASTER_IP} --master_port=${MASTER_PORT} --node_rank=${RANK} \
        ssseg/test.py --nproc_per_node ${AZUREML_NUM_GPUS} --cfgfilepath $CFGFILEPATH --ckptspath $CKPTSPATH ${@:4}
else
    python -m torch.distributed.launch \
        --nnodes=${NODE_COUNT} \
        --node_rank=${RANK} \
        --master_addr=${MASTER_IP} \
        --nproc_per_node=${AZUREML_NUM_GPUS} \
        --master_port=${MASTER_PORT} \
        ssseg/test.py --nproc_per_node ${AZUREML_NUM_GPUS} \
                    --cfgfilepath $CFGFILEPATH \
                    --ckptspath $CKPTSPATH ${@:4}
fi