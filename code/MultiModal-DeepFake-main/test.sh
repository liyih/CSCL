EXPID=CSCL

HOST='localhost'
PORT='1'

NUM_GPU=1

python test.py \
--config 'configs/test.yaml' \
--output_dir './results' \
--launcher pytorch \
--rank 0 \
--log_num ${EXPID} \
--dist-url tcp://localhost:23459 \
--world_size $NUM_GPU \
--test_epoch 49 \
