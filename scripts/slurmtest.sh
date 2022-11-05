#!/usr/bin/env bash
THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $THIS_DIR
cd ..

set -x

PARTITION=$1
JOBNAME=$2
NGPUS=$3
CFGFILEPATH=$4
CHECKPOINTSPATH=$5
GPUSPERNODE=${GPUSPERNODE:-8}
CPUSPERTASK=${CPUSPERTASK:-10}
SRUNAGRS=${SRUNAGRS:-""}
PYARGS=${@:6}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOBNAME} \
    --gres=gpu:${GPUSPERNODE} \
    --ntasks=${NGPUS} \
    --ntasks-per-node=${GPUSPERNODE} \
    --cpus-per-task=${CPUSPERTASK} \
    --kill-on-bad-exit=1 \
    ${SRUNAGRS} \
    python -u ssseg/test.py \
        --nproc_per_node $NGPUS \
        --cfgfilepath $CFGFILEPATH \
        --checkpointspath $CHECKPOINTSPATH \
        --slurm ${PYARGS}