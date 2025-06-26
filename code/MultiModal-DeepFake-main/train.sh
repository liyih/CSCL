EXPID=$(date +"%Y%m%d_%H%M%S")

HOST='localhost'
PORT='1'

NUM_GPU=8
python train.py
--config 'configs/train.yaml' \
--output_dir './results' \
--launcher pytorch \
--rank 0 \
--log_num ${EXPID} \
--dist-url tcp://localhost:23459 \
--world_size $NUM_GPU \
