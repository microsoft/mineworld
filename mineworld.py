import os
import sys
sys.path.append(os.getcwd())
import gradio as gr
from PIL import Image
import numpy as np
import torch
import cv2
from utils import load_model
from omegaconf import OmegaConf
from argparse import ArgumentParser
from collections import deque
import tempfile
import atexit
from torchvision import transforms
from einops import rearrange
from mcdataset import MCDataset
import itertools

class Buttons:
    ATTACK = "attack"
    BACK = "back"
    FORWARD = "forward"
    JUMP = "jump"
    LEFT = "left"
    RIGHT = "right"
    SNEAK = "sneak"
    SPRINT = "sprint"
    USE = "use"
    DROP = "drop"
    SWAPHANDS = "swapHands"
    PICKITEM = "pickItem"

    ALL = [
        ATTACK,
        USE,

        FORWARD,
        BACK,
        LEFT,
        RIGHT,

        JUMP,
        SNEAK,
        SPRINT,

        DROP,
        SWAPHANDS,
        PICKITEM,

        # INVENTORY,
        # ESC,
    ] + [f"hotbar.{i}" for i in range(1, 10)]

KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.s" :"back",
    "key.keyboard.q" :"drop",
    "key.keyboard.w" :"forward",
    "key.keyboard.1" :"hotbar.1",
    "key.keyboard.2" :"hotbar.2",
    "key.keyboard.3" :"hotbar.3",
    "key.keyboard.4" :"hotbar.4",
    "key.keyboard.5" :"hotbar.5",
    "key.keyboard.6" :"hotbar.6",
    "key.keyboard.7" :"hotbar.7",
    "key.keyboard.8" :"hotbar.8",
    "key.keyboard.9" :"hotbar.9",
    "key.keyboard.space" :"jump",
    "key.keyboard.a" :"left",
    "key.keyboard.d" :"right",
    "key.keyboard.left.shift" :"sneak",
    "key.keyboard.left.control" :"sprint",
    "key.keyboard.f" :"swapHands",
}
# Template action
NOOP_ACTION = {
    "forward": 0,
    "back": 0,
    "left": 0,
    "right": 0,

    "jump": 0,
    "attack": 0,
    "use": 0,
    "pickItem": 0,

    "drop": 0,
    "sneak": 0,
    "sprint": 0,
    "swapHands": 0,

    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "camera": np.array([0, 0]),  
}

ACTION_BUTTON = {
    "forward": 0,
    "back": 0,
    "left": 0,
    "right": 0,

    "attack": 0,
    "sprint": 0,
    "jump": 0,
    "use": 0,

    "drop": 0,
    "hotbar.1": 0,
    "pickItem": 0,
}
FOR_BACK = {
    "forward": 0,
    "back": 0,
}
L_R = {
    "left": 0,
    "right": 0,   
}
ATT_USE_DROP = {
    "attack": 0,
    "use": 0,
    "drop": 0,
}
JUMP_SPR = {
    "jump": 0,
    "sprint": 0,
}
HORBAR = {
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,   
}


safe_globals = {"array": np.array}

AGENT_RESOLUTION = (384, 224)
CAMERA_SCALER = 360.0 / 2400.0
TOKEN_PER_IMAGE = 336
TOKEN_PER_ACTION = 11
VIDEO_FRAMES = []
GENERATED_FILES = []
frame_cache = []
action_cache = []
last_pos = 0
MC_ACTION_MAP = MCDataset()
SHOW_FRAMES = 8
REFERENCE_FRAME = None
CONTEXT_LEN = None
DIAGD = False
WINDOWSIZE = 4

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--scene', type=str, default='./assets/scene.mp4')
    parser.add_argument('--model_ckpt', type=str, default='./checkpoints/700M_16f.pt')
    parser.add_argument('--config', type=str, default='./configs/700M_16f.yaml')
    parser.add_argument('--reference_frame', type=int,  default=8)
    parser.add_argument('--diagd', action='store_true', help='use diagd')
    parser.add_argument('--window_size', type=int,  default=4)
    args = parser.parse_args()
    return args

def make_action_dict(action_line):
    action_dict = {'ESC': 0, 'back': 0, 'drop': 0, 'forward': 0, 'hotbar.1': 0, 'hotbar.2': 0, 'hotbar.3': 0, 'hotbar.4': 0, 'hotbar.5': 0, 'hotbar.6': 0, 'hotbar.7': 0, 'hotbar.8': 0, 'hotbar.9': 0, 'inventory': 0, 'jump': 0, 'left': 0, 'right': 0, 'sneak': 0, 'sprint': 0, 'swapHands': 0, 'camera': np.array([0, 0]), 'attack': 0, 'use': 0, 'pickItem': 0}
    if isinstance(action_line, str):
        action_line = action_line.split(",")
        action_dict['camera'] = np.array((int(action_line[-2]), int(action_line[-1])))
    for act in action_line:
        if act in Buttons.ALL:
            action_dict[act] = 1
    
    return action_dict

def stack_images(imgs):
    width, height = imgs[0].size
    new_im = Image.new('RGB', (4*width, height*2))
    for i, im in enumerate(imgs):
        new_im.paste(im, (width*(i%4), height*(i//4)))
    return new_im

def get_action_line(acts):
    action_lst = []
    for k in acts.keys():
        if k != "camera" and acts[k] == 1:
            action_lst.append(k)
    action_lst.append(str(acts["camera"][0]))
    action_lst.append(str(acts["camera"][1]))    

    return ",".join(action_lst)

def run_prediction(btns_choices, cam_x_input, cam_y_input):
    global frame_cache, action_cache, actions_show, images_show, VIDEO_FRAMES, last_pos, CONTEXT_LEN, REFERENCE_FRAME    
    assert len(frame_cache) == len(action_cache)+1
    if len(action_cache) >= CONTEXT_LEN - 1:
        for _ in range(CONTEXT_LEN - REFERENCE_FRAME):
            frame_cache.popleft()
            action_cache.popleft() 
        model.transformer.refresh_kvcache()
        _frame_iter = itertools.islice(frame_cache, 0, len(frame_cache)-1)
        _act_iter = itertools.islice(action_cache, 0, len(action_cache))
        _vis_act = [
                torch.cat([img, act], dim=1)
                for img, act in zip(_frame_iter, _act_iter)
            ]
        _vis_act.append(frame_cache[-1])
        _vis_act = torch.cat(_vis_act, dim=-1)
        _, last_pos = model.transformer.prefill_for_gradio(_vis_act)
        

    act_dict = make_action_dict(btns_choices)
    act_dict['camera'] = np.array((int(cam_y_input), int(cam_x_input)))
    ongoing_act = MC_ACTION_MAP.get_action_index_from_actiondict(act_dict, action_vocab_offset=8192)
    ongoing_act = torch.tensor(ongoing_act).unsqueeze(0).to("cuda")
    action_cache.append(ongoing_act)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        if DIAGD:
            next_frame, last_pos = model.transformer.diagd_img_token_for_gradio(input_action=ongoing_act, position_id = last_pos, max_new_tokens=TOKEN_PER_IMAGE, windowsize=4)
        else:
            next_frame, last_pos = model.transformer.decode_img_token_for_gradio(input_action=ongoing_act, position_id = last_pos, max_new_tokens=TOKEN_PER_IMAGE + 1) # +1 to fill kvcache
        
    last_pos = last_pos[0]
    next_frame = torch.cat(next_frame, dim=-1).to("cuda")
    frame_cache.append(next_frame)
    next_frame = tokenizer.token2image(next_frame)
    next_frame = Image.fromarray(next_frame)
    if len(images_show) >= SHOW_FRAMES:
        images_show.popleft()
        actions_show.popleft()
    btns_choices = btns_choices + [np.array((int(cam_y_input), int(cam_x_input)))]
    actions_show.append(','.join(str(x) for item in btns_choices for x in (item if isinstance(item, np.ndarray) else [item])))
    images_show.append(next_frame)
    VIDEO_FRAMES.append(next_frame)

    return next_frame, stack_images(images_show), "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;".join([str(x) for x in actions_show])

def run_prediction_n_times(n, btns_1, btns_2, btns_3, btns_4, btns_5, cam_x_input, cam_y_input):
    btns_choices = btns_1 + btns_2 + btns_3 + btns_4 + btns_5
    if cam_x_input is None:
        cam_x_input = 0
    if cam_y_input is None:
        cam_y_input = 0
    if n is None:
        n = 1
    for i in range(n):
        yield run_prediction(btns_choices, cam_x_input, cam_y_input)

def step_pred_source_video_right(video_path, start):
    global VIDEO_FRAMES, frame_cache, action_cache, REFERENCE_FRAME, CONTEXT_LEN, REFERENCE_FRAME
    VIDEO_FRAMES.clear(); frame_cache.clear(); action_cache.clear()
    if start is None or start < 0 or start > MAX_FRAME:
        start = 0
    return step_video(video_path, start, REFERENCE_FRAME)

def on_download_button_click(fps=6):
    if not VIDEO_FRAMES:
        print("The frames list is empty.")
        return

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir="/tmp")
    video_path = temp_file.name
    temp_file.close()

    video_writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        AGENT_RESOLUTION
    )

    for frame in VIDEO_FRAMES:
        frame_np = np.array(frame)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

    video_writer.release()
    GENERATED_FILES.append(video_path)
    os.chmod(video_path, 0o644)
    return video_path

def cleanup_files():
    for video_path in GENERATED_FILES:
        try:
            os.remove(video_path)
            print(f"Deleted file: {video_path}")
        except OSError as e:
            print(f"Error deleting file {video_path}: {e}")

atexit.register(cleanup_files)

def step_video(video_path, start_frame, frame_count):
    global images_show, actions_show, frame_cache, action_cache, VIDEO_FRAMES, last_pos, CONTEXT_LEN, REFERENCE_FRAME
    VIDEO_FRAMES = []
    images_show = []
    actions_show = []
    video = cv2.VideoCapture(video_path)
    json_data = MC_ACTION_MAP.read_jsonl(video_path[:-4]+".jsonl")

    frames_tensor = []
    action_cache = []
    for i in range(start_frame, start_frame + frame_count):
        step_action = json_data[i]
        step_action, _ = MC_ACTION_MAP.json_action_to_env_action(step_action)
        actions_show.append(get_action_line(step_action))
        
        act_index = MC_ACTION_MAP.get_action_index_from_actiondict(step_action, action_vocab_offset=8192)
        act_index = torch.tensor(act_index).unsqueeze(0)
        action_cache.append(act_index.to("cuda"))

        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        try:
            if not ret:
                raise ValueError(f"frame {i} not ret")
            cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)

            frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
            frame = cv2.resize(frame, AGENT_RESOLUTION, interpolation=cv2.INTER_LINEAR)
            images_show.append(Image.fromarray(frame))
            VIDEO_FRAMES.append(Image.fromarray(frame))
            frames_tensor.append(torch.from_numpy(frame))
            
        except Exception as e:
            print(f"Could not read frame from video {video_path}: {e}")
    video.release()
    frames_tensor = torch.stack(frames_tensor, dim=0).to("cuda")
    frames_tensor = frames_tensor.permute(0, 3, 1, 2).float() / 255.0
    frames_tensor = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(frames_tensor)

    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        images_token = tokenizer.tokenize_images(frames_tensor)
    images_token = rearrange(images_token, '(b t) h w -> (b t) (h w)', b=1)
    
    frame_cache = deque(torch.split(images_token, split_size_or_sections=1, dim=0))
    action_cache = deque(action_cache)
    action_cache.pop()

    images_show = deque(images_show)
    actions_show = deque(actions_show)

    actions_show.pop()

    
    model.transformer.refresh_kvcache()
    _frame_iter = itertools.islice(frame_cache, 0, len(frame_cache)-1)
    _act_iter = itertools.islice(action_cache, 0, len(action_cache))

    
    _vis_act = [
            torch.cat([img, act], dim=1)
            for img, act in zip(_frame_iter, _act_iter)
        ]
    _vis_act.append(frame_cache[-1])
    _vis_act = torch.cat(_vis_act, dim=-1)

    _, last_pos = model.transformer.prefill_for_gradio(_vis_act)

    while len(images_show) > SHOW_FRAMES:
        images_show.popleft()
        actions_show.popleft()
        # WARNING: why dont pop actions
    
    return stack_images(images_show), "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;".join([str(x) for x in actions_show]), None


css = """
.custom-tab h2 { 
    font-size: 34px;  /* 字体大小 */
    font-weight: bold;  /* 加粗字体 */
    color: #ff6600;  /* 字体颜色 */
    text-shadow: 1px 1px 2px #000000; /* 文字阴影效果 */
}
"""

if __name__ == "__main__":
    args = get_args()
    if args.diagd:
        DIAGD = True
        WINDOWSIZE = args.window_size
    cap = cv2.VideoCapture(args.scene)
    global MAX_FRAME
    MAX_FRAME = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 10

    config = OmegaConf.load(args.config)
    REFERENCE_FRAME = args.reference_frame
    CONTEXT_LEN = int(config.model.params.transformer_config.params.max_position_embeddings / (TOKEN_PER_ACTION + TOKEN_PER_IMAGE))
    assert CONTEXT_LEN > REFERENCE_FRAME
    model = load_model(config, args.model_ckpt, gpu=True, eval_mode=True)
    tokenizer = model.tokenizer
    with gr.Blocks(css=css) as demo:
        with gr.Tab("MineWorld", elem_classes="custom-tab"):
            source_video_path = gr.Text(value=args.scene, visible=False)
            with gr.Row():
                source_video_actions = gr.Markdown(visible=False)
                instruction = gr.Markdown("press 'Jump to start frame' to init or restart the game, you can choose different sences by modifed start frame from 0 to 4100", visible=False)
            with gr.Row():
                source_video_images = gr.Image(width=1280, height=360, label="last 8 frames", show_fullscreen_button = True, every=1)

            with gr.Row(equal_height=True):                    
                with gr.Column(min_width=60):
                    vid_frame_start = gr.Number(step=1, value=0, info="start frame", min_width=20, show_label=False, minimum=0, maximum=MAX_FRAME)
                        # vid_num_frames = gr.Number(step=1, value=4, label="num_frames", min_width=50)
                # with gr.Column(min_width=60):
                    run_steps = gr.Number(step=1, value=1, info="Repeat same action n times", min_width=20, minimum=1, maximum=8, show_label=False)
                with gr.Column(min_width=60):
                    btn1 = list(FOR_BACK.keys())
                    btns_1 = gr.CheckboxGroup(choices=btn1, show_label=False)
                    vid_right_btn = gr.Button(value="Jump to start frame", size='sm')
                with gr.Column(min_width=60):
                    btn2 = list(L_R.keys())
                    btns_2 = gr.CheckboxGroup(choices=btn2, show_label=False)
                    predict_run_btn = gr.Button(value="Run", variant="primary", size='sm')
                with gr.Column(min_width=60):
                    btn4 = list(JUMP_SPR.keys())
                    btns_4 = gr.CheckboxGroup(choices=btn4, show_label=False)
                    download_game_btn = gr.Button("Generate Video", size='sm')
                with gr.Column(min_width=60):
                    cam_y_input = gr.Number(step=1, value=0, info="camera Y ⬆️(-),0,(+)⬇️", min_width=20, minimum=-90, maximum=90, show_label=False)
                    cam_x_input = gr.Number(step=1, value=0, info="camera X ⬅️(-),0,(+)➡️", min_width=20, minimum=-90, maximum=90, show_label=False)
                
                    
            with gr.Row():
                with gr.Column(min_width=250):
                    video_display = gr.Video(label="video", width=384, height=224)
                with gr.Column(min_width=200): 
                    predict_result_imgs = gr.Image(label="last generated frame",width=384, height=224)
                with gr.Column(min_width=200):
                    with gr.Row():
                        btn3 = list(ATT_USE_DROP.keys())
                        btns_3 = gr.CheckboxGroup(choices=btn3, show_label=False)
                    with gr.Row():
                        btn5 = list(HORBAR.keys())
                        btns_5 = gr.CheckboxGroup(choices=btn5, show_label=False)

                    
                
            vid_right_btn.click(fn=step_pred_source_video_right, inputs=[source_video_path, vid_frame_start],
                                outputs=[source_video_images, source_video_actions, predict_result_imgs])                
            predict_run_btn.click(fn=run_prediction_n_times, inputs=[run_steps, btns_1, btns_2, btns_3, btns_4, btns_5, cam_x_input, cam_y_input],
                                    outputs=[predict_result_imgs, source_video_images, source_video_actions],)
            download_game_btn.click(fn=on_download_button_click, inputs=[], outputs=video_display)
            demo.load(fn=step_pred_source_video_right, inputs=[source_video_path, gr.Number(value=25, visible=False)],
                                outputs=[source_video_images, source_video_actions, predict_result_imgs])
    demo.queue()
    
    demo.launch(server_name="0.0.0.0", max_threads=256, server_port=7861, share=True)
