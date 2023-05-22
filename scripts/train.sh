EXPID=$1
CONFIG=$2
PORT=${3:-1888}

HOST=$(hostname -i)


python ./scripts/train_smpl_cam.py \
    --cfg ${CONFIG} \
    --exp-id ${EXPID} \
    --num_threads 0 \
    --snapshot 2 \
    --rank 0 \
    --dist-url tcp://${HOST}:${PORT} \
    --no_dist


### distributed training ###
# python ./scripts/train_smpl_cam.py \
#     --cfg ${CONFIG} \
#     --exp-id ${EXPID} \
#     --num_threads 8 \
#     --snapshot 2 \
#     --rank ${RANK} \
#     --dist-url tcp://${MASTER_ADDR}:${MASTER_PORT}