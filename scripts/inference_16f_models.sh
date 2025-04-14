DATA_ROOT="validation/validation"
###########################
#### Inference 1200M models 
###########################
CONFIG="configs/1200M_16f.yaml"
CKPT_PATH="checkpoints/1200M_16f.ckpt"
OUTPUT_PATH="./videos/1200M_16f200_demo1gen15_naive"
python inference.py \
        --data_root $DATA_ROOT  \
        --config $CONFIG \
        --model_ckpt $CKPT_PATH \
        --demo_num 1 --frames 15  \
        --accelerate-algo 'naive' \
        --top_p 0.8 \
        --output_dir  $OUTPUT_PATH