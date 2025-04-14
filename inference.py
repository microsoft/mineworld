import os
import cv2
import torch
import time
import numpy as np
from tqdm import tqdm
from rich import print
from PIL import Image
from pathlib import Path
from torch import autocast
from einops import rearrange
from mcdataset import MCDataset
from omegaconf import OmegaConf
from torchvision import transforms
from argparse import ArgumentParser
from utils import load_model, tensor_to_uint8
torch.backends.cuda.matmul.allow_tf32 = False

ACCELERATE_ALGO = [
    'naive','image_diagd'
]

TARGET_SIZE=(224,384)
TOKEN_PER_IMAGE = 347 # IMAGE = PIX+ACTION
TOKEN_PER_PIX = 336

safe_globals = {"array": np.array}


def token2video(code_list, tokenizer, save_path, fps, device = 'cuda'):
    """
    change log:  we don't perform path processing inside functions to enable extensibility
    save_path: str, path to save the video, expect to endwith .mp4
    
    """
    if len(code_list) % TOKEN_PER_PIX != 0:
        print(f"code_list length {len(code_list)} is not multiple of {TOKEN_PER_PIX}")
        return
    num_images = len(code_list) // TOKEN_PER_PIX
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, fps, (384, 224))
    for i in range(num_images):
        code = code_list[i*TOKEN_PER_PIX:(i+1)*TOKEN_PER_PIX]
        code = torch.tensor([int(x) for x in code], dtype=torch.long).to(device)
        img = tokenizer.token2image(code) # pixel
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(frame)
    video.release()

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--model_ckpt', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--demo_num', type=int, default=1)
    parser.add_argument('--frames', type=int, required=True)
    parser.add_argument('--window_size', type=int, default=2)
    parser.add_argument('--accelerate-algo', type=str, default='naive', help=f"Accelerate Algorithm Option: {ACCELERATE_ALGO}")
    parser.add_argument('--fps', type=int, default=6)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--top_k', type=int, help='Use top-k sampling')
    group.add_argument('--top_p', type=float, help='Use top-p (nucleus) sampling')
    parser.add_argument('--val_data_num', type=int, default=500, help="number of validation data")
    args = parser.parse_args()
    return args


def lvm_generate(args, model, output_dir, demo_video):
    """
    """
    ### 1. set video input/output path
    input_mp4_path = os.path.join(args.data_root, demo_video)
    input_action_path = os.path.join(args.data_root, demo_video.replace('mp4','jsonl'))

    output_mp4_path = str(output_dir / demo_video)
    output_action_path = output_mp4_path.replace('.mp4', '.jsonl')
    # backup action 
    os.system(f"cp {input_action_path} {output_action_path}")
    if os.path.exists(output_mp4_path):
        print(f"output path {output_mp4_path} exist")
        return {}
    
    device = model.transformer.device
    ### 2. load action into list 
    action_list = []
    mcdataset = MCDataset()
    with open(input_action_path, 'r') as f:
        for line in f:
            line = eval(line.strip(), {"__builtins__": None}, safe_globals)
            line['camera'] = np.array(line['camera'])
            act_index = mcdataset.get_action_index_from_actiondict(line, action_vocab_offset=8192)
            action_list.append(act_index)
    ### 3. load video frames 
    cap = cv2.VideoCapture(input_mp4_path)
    start_frame = 0
    end_frame = args.demo_num
    frames = []
    for frame_idx in range(start_frame, end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Error in reading frame {frame_idx}")
            continue
        cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
        frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
        frame = torch.from_numpy(frame)
        frames.append(frame)
    frames = torch.stack(frames, dim=0).to(device)
    frames = frames.permute(0, 3, 1, 2)
    frames = frames.float() / 255.0
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    frames = normalize(frames)
    
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        img_index = model.tokenizer.tokenize_images(frames)
    img_index = rearrange(img_index, '(b t) h w -> b t (h w)', b=1)
     
    all_generated_tokens = []

    action_all = action_list[end_frame: end_frame + args.frames]
    action_all = torch.tensor(action_all).unsqueeze(1).to(device)
    image_input = rearrange(img_index, 'b t c -> b (t c)')
    

    start_t = time.time() 
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        if args.accelerate_algo == 'naive':
            outputs = model.transformer.naive_generate( input_ids=image_input, max_new_tokens=TOKEN_PER_PIX*args.frames, action_all=action_all, top_k=args.top_k, top_p=args.top_p)
        elif args.accelerate_algo == 'image_diagd':
            outputs = model.transformer.img_diagd_generate(input_ids=image_input, max_new_tokens=TOKEN_PER_PIX*args.frames, action_all=action_all,windowsize = args.window_size, top_k=args.top_k, top_p=args.top_p)
        else:
            raise ValueError(f"Unknown accelerate algorithm {args.accelerate_algo}")
    end_t = time.time()
    all_generated_tokens.extend(outputs.tolist()[0])
    new_length = len(all_generated_tokens)
    time_costed = end_t - start_t 
    token_per_sec = new_length / time_costed
    frame_per_sec = token_per_sec / TOKEN_PER_PIX
    print(f"{new_length} token generated; cost {time_costed:.3f} second; {token_per_sec:.3f} token/sec {frame_per_sec:.3f} fps")
    token2video(all_generated_tokens, model.tokenizer, str(output_path / demo_video), args.fps, device)  
    # return for evaluation 
    return_item = {
        "time_costed": time_costed,
        "token_num": new_length,
    }
    return return_item
if __name__ == '__main__':
    args = get_args()
    config = OmegaConf.load(args.config)
    output_path = Path(args.output_dir)
    precision_scope = autocast
    os.makedirs(output_path, exist_ok=True)

    model = load_model(config, args.model_ckpt, gpu=True, eval_mode=True)
    print(f"[bold magenta][MINEWORLD][INFERENCE][/bold magenta] Load Model From {args.model_ckpt}")
    # get accelearte algoritm
    args.accelerate_algo = args.accelerate_algo.lower()
    if args.accelerate_algo not in ACCELERATE_ALGO:
        print(f"[bold red][Warning][/bold red] {args.accelerate_algo} is not in {ACCELERATE_ALGO}, use naive")
        args.accelerate_algo = 'naive'
    num_item = 0
    for root, _, files in os.walk(args.data_root):
        files = [f for f in files  if f.endswith('.mp4')] # mp4 would not influence progress bar 
        files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        for file in tqdm(files):
            return_item = lvm_generate(args, model, output_path,file)
            num_item += 1
            if num_item  >= args.val_data_num:
                print(f"[bold magenta][MINEWORLD][INFERENCE][/bold magenta]  reach val data num limit {args.val_data_num}")
                break