import sys
import os
sys.path.append(os.getcwd())
from common_metrics_on_video_quality.calculate_fvd import calculate_fvd
from common_metrics_on_video_quality.calculate_lpips import calculate_lpips
from common_metrics_on_video_quality.calculate_ssim import calculate_ssim
from common_metrics_on_video_quality.calculate_psnr import calculate_psnr
import os
import cv2
import torch
import numpy as np
import argparse
import json
device = torch.device("cuda")

def load_videos_to_tensor(video_dir, number_of_videos, video_length, channel, size,video_files=None):
    videos_tensor = torch.zeros(number_of_videos, video_length, channel, size[0], size[1], requires_grad=False)
    if video_files is None:
        video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4'))]
    video_files = sorted(video_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    video_files = video_files[:number_of_videos]
    for i, video_file in enumerate(video_files):
        video_path = os.path.join(video_dir, video_file)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            continue

        frames = []
        # get video total length ; our gt has 16 frame but we only use 15 frame so set video_length to 15 and start frame to 1 
        real_video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if real_video_length > video_length:
            # set start frame to video_length - video_length
            start_frame = real_video_length - video_length
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        else:
            # set start frame to 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while len(frames) < video_length:
            ret, frame = cap.read()
            if not ret: 
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (size[1], size[0]))  # Resize to (height, width)
            frames.append(frame)
        if len(frames) < video_length:
            print(f"Video {video_file} has fewer frames than expected. Expected: {video_length}, Found: {len(frames)} Exiting...")
            exit(1)
        cap.release()
        frames_np = np.array(frames[:video_length]) 
        frames_np = np.transpose(frames_np, (0, 3, 1, 2)) 
        videos_tensor[i] = torch.tensor(frames_np, dtype=torch.float32) / 255.0 

    return videos_tensor

# python scripts/tvideo/mc/common_metrics.py --video_dir1 metrics_table_1/oasis/oasis_official_results_no_demo_1gen15_mineworld_curation --video_dir2 metrics_table_1/frame_16_curation --video_length 15 --channel 3 --size "(224,384)" --output-file test_metrics.json 
def main():
    parser = argparse.ArgumentParser(description="Calculate FVD for two sets of videos.")
    parser.add_argument("--video_dir1", type=str, required=True, help="Path to the first directory containing videos.")
    parser.add_argument("--video_dir2", type=str, required=True, help="Path to the second directory containing videos.")
    parser.add_argument("--video_length", type=int, default=32, help="Number of frames to retain from each video.")
    parser.add_argument("--channel", type=int, default=3, help="Number of channels in the videos (default: 3 for RGB).")
    parser.add_argument("--size", type=str, default="(224,384)", help="Size of the video frames (default: 256x256).")
    parser.add_argument("--output-file", type=str)
    args = parser.parse_args()
    args.size = eval(args.size)
    print("args.size", args.size)
    number_of_videos = len([f for f in os.listdir(args.video_dir1) if f.endswith(".mp4")])
    video_files = [f for f in os.listdir(args.video_dir1) if f.endswith(('.mp4'))]
    number_of_videos = min(500,len(video_files))
    print("number_of_videos", number_of_videos)
    videos1 = load_videos_to_tensor(args.video_dir1, number_of_videos, args.video_length, args.channel, args.size, video_files)
    videos2 = load_videos_to_tensor(args.video_dir2, number_of_videos, args.video_length, args.channel, args.size, video_files)
    print("videos1.shape", videos1.shape, "videos2.shape", videos2.shape)
    device = torch.device("cuda")

    print(args.output_file)
    result = {}
    result['fvd'] = calculate_fvd(videos1, videos2, device, method='styleganv')
    # result['fvd'] = calculate_fvd(videos1, videos2, device, method='videogpt')
    result['ssim'] = calculate_ssim(videos1, videos2)
    result['psnr'] = calculate_psnr(videos1, videos2)
    result['lpips'] = calculate_lpips(videos1, videos2, device)
    lpips_mean =  np.mean(list(result['lpips']['value']))
    ssim_mean =  np.mean(list(result['ssim']['value']))
    psnr_mean =  np.mean(list(result['psnr']['value']))
    fvd_mean =  np.mean(list(result['fvd']['value']))
    data_item = {"exp_name":args.video_dir1, "fvd":fvd_mean, "lpips":lpips_mean, "ssim":ssim_mean, "psnr":psnr_mean}
    print(data_item)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    result['mean'] = data_item
    with open(args.output_file, "w") as f:
        json.dump(result, f, indent=4)
    print("results saved to ", args.output_file)

if __name__ == "__main__":
    main()