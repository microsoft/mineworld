#!/bin/bash

VIDEO_RESULTS_ROOT_DEFAULT="videos"
METRICS_ROOT_DEFAULT="metrics_log"
JSONL_PATH_DEFAULT="validation/validation"
IDM_CKPT_DIR="checkpoints/IDM"

VIDEO_RESULTS_ROOT=${1:-$VIDEO_RESULTS_ROOT_DEFAULT}
METRICS_ROOT=${2:-$METRICS_ROOT_DEFAULT}
JSONL_PATH=${3:-$JSONL_PATH_DEFAULT}

echo "VIDEO_RESULTS_ROOT = $VIDEO_RESULTS_ROOT"
echo "METRICS_ROOT       = $METRICS_ROOT"
echo "JSONL_PATH         = $JSONL_PATH"

# Loop through each subdirectory in VIDEO_RESULTS_ROOT
for video_dir1 in "$VIDEO_RESULTS_ROOT"/*/; do
    # Skip the 'metrics' directory
    if [ -d "$video_dir1" ] && [ "$(basename "$video_dir1")" != "metrics" ]; then
        # Construct the output file name based on the video directory name
        fvd_output_file="$METRICS_ROOT/fvd_$(basename "$video_dir1").json"
        echo $fvd_output_file
        # Run the python command for each video directory
        python metrics/common_metrics.py --video_dir2 $JSONL_PATH --video_length 15 --channel 3 --size "(224,384)" \
                --video_dir1 "$video_dir1" --output-file "$fvd_output_file"

        idm_output_file="$METRICS_ROOT/idm_$(basename "$video_dir1").json"
        python metrics/IDM/inverse_dynamics_model.py --weights $IDM_CKPT_DIR/"4x_idm.weights" \
            --infer-demo-num 1 --n-frames 15 \
            --model $IDM_CKPT_DIR/"4x_idm.model" --video-path $video_dir1 \
            --output-file "$idm_output_file" \
            --jsonl-path $JSONL_PATH
    fi
done

python  metrics/tabulate_all_results.py --input_dir $METRICS_ROOT --output_path $METRICS_ROOT/latest_metrics.csv