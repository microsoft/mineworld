# Borrowed from VPT (https://github.com/openai/Video-Pre-Training)

import numpy as np
import torch as th
import cv2
import pdb
from gym3.types import DictType
from gym import spaces
from tqdm import tqdm

from lib.action_mapping import CameraHierarchicalMapping, IDMActionMapping
from lib.actions import ActionTransformer
from lib.policy import InverseActionPolicy
from lib.torch_util import default_device_type, set_default_torch_device

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Hardcoded settings
AGENT_RESOLUTION = (128, 128)
safe_globals = {"array": np.array}
def resize_image(img, target_resolution):
    # For your sanity, do not resize with any function than INTER_LINEAR
    img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)
    return img


ACTION_TRANSFORMER_KWARGS = dict(
    camera_binsize=2,
    camera_maxval=10,
    camera_mu=10,
    camera_quantization_scheme="mu_law",
)

class IDMAgent:
    """
    Sugarcoating on the inverse dynamics model (IDM) used to predict actions Minecraft players take in videos.

    Functionally same as MineRLAgent.
    """
    def __init__(self, idm_net_kwargs, pi_head_kwargs, device=None):
        if device is None:
            device = default_device_type()
        self.device = th.device(device)
        # Set the default torch device for underlying code as well
        set_default_torch_device(self.device)
        self.action_mapper = IDMActionMapping(n_camera_bins=11)
        action_space = self.action_mapper.get_action_space_update()
        action_space = DictType(**action_space)

        self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

        idm_policy_kwargs = dict(idm_net_kwargs=idm_net_kwargs, pi_head_kwargs=pi_head_kwargs, action_space=action_space)

        self.policy = InverseActionPolicy(**idm_policy_kwargs).to(device)
        self.hidden_state = self.policy.initial_state(1)
        self._dummy_first = th.from_numpy(np.array((False,))).to(device)

    def load_weights(self, path):
        """Load model weights from a path, and reset hidden state"""
        self.policy.load_state_dict(th.load(path, map_location=self.device), strict=False)
        self.reset()

    def reset(self):
        """Reset agent to initial state (i.e., reset hidden state)"""
        self.hidden_state = self.policy.initial_state(1)

    def _video_obs_to_agent(self, video_frames):
        imgs = [resize_image(frame, AGENT_RESOLUTION) for frame in video_frames]
        # Add time and batch dim
        imgs = np.stack(imgs)[None]
        agent_input = {"img": th.from_numpy(imgs).to(self.device)}
        return agent_input

    def _agent_action_to_env(self, agent_action):
        """Turn output from policy into action for MineRL"""
        # This is quite important step (for some reason).
        # For the sake of your sanity, remember to do this step (manual conversion to numpy)
        # before proceeding. Otherwise, your agent might be a little derp.
        action = {
            "buttons": agent_action["buttons"].cpu().numpy(),
            "camera": agent_action["camera"].cpu().numpy()
        }
        minerl_action = self.action_mapper.to_factored(action)
        minerl_action_transformed = self.action_transformer.policy2env(minerl_action)
        return minerl_action_transformed

    def predict_actions(self, video_frames):
        """
        Predict actions for a sequence of frames.

        `video_frames` should be of shape (N, H, W, C).
        Returns MineRL action dict, where each action head
        has shape (N, ...).

        Agent's hidden state is tracked internally. To reset it,
        call `reset()`.
        """
        agent_input = self._video_obs_to_agent(video_frames)
        # The "first" argument could be used to reset tell episode
        # boundaries, but we are only using this for predicting (for now),
        # so we do not hassle with it yet.
        dummy_first = th.zeros((video_frames.shape[0], 1)).to(self.device)
        predicted_actions, self.hidden_state, _ = self.policy.predict(
            agent_input, first=dummy_first, state_in=self.hidden_state,
            deterministic=True
        )
        predicted_minerl_action = self._agent_action_to_env(predicted_actions)
        return predicted_minerl_action
# NOTE: this is _not_ the original code of IDM!
# As such, while it is close and seems to function well,
# its performance might be bit off from what is reported
# in the paper.
import os 
from argparse import ArgumentParser
import pickle
import cv2
import numpy as np
import json
import torch as th
ENV_KWARGS = dict(
    fov_range=[70, 70],
    frameskip=1,
    gamma_range=[2, 2],
    guiscale_range=[1, 1],
    resolution=[640, 360],
    cursor_size_range=[16.0, 16.0],
)


KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape" :"ESC",
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
    "key.keyboard.e" :"inventory",
    "key.keyboard.space" :"jump",
    "key.keyboard.a" :"left",
    "key.keyboard.d" :"right",
    "key.keyboard.left.shift" :"sneak",
    "key.keyboard.left.control" :"sprint",
    "key.keyboard.f" :"swapHands",
}

# Template action
NOOP_ACTION = {
    "ESC": 0,
    "back": 0,
    "drop": 0,
    "forward": 0,
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "inventory": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0,
    "swapHands": 0,
    "camera": np.array([0, 0]),
    "attack": 0,
    "use": 0,
    "pickItem": 0,
}


# Matches a number in the MineRL Java code regarding sensitivity
# This is for mapping from recorded sensitivity to the one used in the model
CAMERA_SCALER = 360.0 / 2400.0


def json_action_to_env_action(json_action):
    """
    Converts a json action into a MineRL action.
    Returns (minerl_action, is_null_action)
    """
    if "ESC" in json_action:
        return json_action, False
    # This might be slow...
    env_action = NOOP_ACTION.copy()
    # As a safeguard, make camera action again so we do not override anything
    env_action["camera"] = np.array([0, 0])

    is_null_action = True
    keyboard_keys = json_action["keyboard"]["keys"]
    for key in keyboard_keys:
        # You can have keys that we do not use, so just skip them
        # NOTE in original training code, ESC was removed and replaced with
        #      "inventory" action if GUI was open.
        #      Not doing it here, as BASALT uses ESC to quit the game.
        if key in KEYBOARD_BUTTON_MAPPING:
            env_action[KEYBOARD_BUTTON_MAPPING[key]] = 1
            is_null_action = False

    mouse = json_action["mouse"]
    camera_action = env_action["camera"]
    camera_action[0] = mouse["dy"] * CAMERA_SCALER
    camera_action[1] = mouse["dx"] * CAMERA_SCALER

    if mouse["dx"] != 0 or mouse["dy"] != 0:
        is_null_action = False
    else:
        if abs(camera_action[0]) > 180:
            camera_action[0] = 0
        if abs(camera_action[1]) > 180:
            camera_action[1] = 0

    mouse_buttons = mouse["buttons"]
    if 0 in mouse_buttons:
        env_action["attack"] = 1
        is_null_action = False
    if 1 in mouse_buttons:
        env_action["use"] = 1
        is_null_action = False
    if 2 in mouse_buttons:
        env_action["pickItem"] = 1
        is_null_action = False

    return env_action, is_null_action

def load_action_jsonl_old(path_action):
    action_list = []
    with open(path_action, 'r') as f:
        for line in f:
            line = eval(line.strip(), {"__builtins__": None}, safe_globals)
            line['camera'] = np.array(line['camera'])
            # act_dict = vis_act_tok.tokenize_actions(action_dict=line)
            action_list.append(act_dict)  
    return action_list

def load_action_jsonl(json_path):
    with open(json_path) as json_file:
        json_lines = json_file.readlines()
        json_data = "[" + ",".join(json_lines) + "]"
        json_data = json.loads(json_data)
    return json_data 


# loss on frame - avg on video - avg on dataset 
def evaluate_IDM_quality(model, weights,jsonl_folder, video_folder, infer_demo_num, n_frames, output_file):
    """
    Evaluate the quality of a IDM model on a dataset of videos.

    Args:
        video_folder (str): Path to the folder containing videos.
        model (str): Path to the '.model' file to be loaded.
        weights (str): Path to the '.weights' file to be loaded.
        n_batches (int): Number of batches to process.
        n_frames (int): Number of frames to process at a time.
    """
    ## set up IDM model 
    agent_parameters = pickle.load(open(model, "rb"))
    net_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    # pi_head_kwargs["temperature"] = 1.0
    agent = IDMAgent(idm_net_kwargs=net_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)
    # Load video files
    video_files = os.listdir(video_folder)
    video_files = [f for f in video_files if f.endswith(".mp4")]
    video_files = sorted(video_files)
    video_files = [os.path.join(video_folder, f) for f in video_files]
    eval_num = min(500,len(video_files)) 
    video_files = video_files[:eval_num]
    dataset_labels = {}
    for video_file in tqdm(video_files):
        json_file = os.path.join(jsonl_folder,os.path.basename(video_file).replace(".mp4",".jsonl"))
        # old implementation
        # action_loss,video_avg_loss,predicted_actions_list = eval_1_video(agent, video_file, json_file, infer_demo_num, n_frames) 
        
        # load predicted actions and recorded actions
        predicted_actions,recorded_actions = idm_prediction(agent, video_file,json_file, infer_demo_num, n_frames)
        # construct labels
        subtasks_labels = define_exclusive_classification_task(predicted_actions,recorded_actions,calculate_hot_bar = False)
        for key in subtasks_labels:
            if key not in dataset_labels:
                dataset_labels[key] = {"pred_labels":[] , "rec_labels":[], "class_num":0}
            dataset_labels[key]["pred_labels"].append(subtasks_labels[key]["pred_labels"])# array 
            dataset_labels[key]["rec_labels"].append(subtasks_labels[key]["rec_labels"]) # array 
            dataset_labels[key]["class_num"] = subtasks_labels[key]["class_num"]
    
    dataset_results ={}
    for key in dataset_labels:
        pred_labels = np.stack(dataset_labels[key]["pred_labels"]).flatten() # [num_videos , num_frames] -> [video_num x frame_num]
        rec_labels = np.stack(dataset_labels[key]["rec_labels"]).flatten()   # [num_videos , num_frames] -> [video_num x frame_num]
        dataset_results[key]=classification_metric(pred_labels, rec_labels, dataset_labels[key]["class_num"])
    # import pdb;pdb.set_trace()
    metric_mean_on_task = {}
    metrics = ['precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'recall_macro', 'f1_macro']
    tasks = dataset_results.keys()
    for key in metrics:
        if key == "class_num":
            continue
        metric_mean_on_task[key] = np.mean([dataset_results[task][key] for task in tasks])
    dataset_results["metric_mean_on_task"] = metric_mean_on_task
    ## change all keys into str
    dataset_results = {str(k): v for k, v in dataset_results.items()}
        
    print(dataset_results)
    print("===========================================")
    print(f"{output_file} IDM Metric: {metric_mean_on_task}")
    with open(output_file, 'w') as f:
        f.write(json.dumps(dataset_results,indent=4) + "\n")

def construct_classification_labels(idm_actions:dict[str, list[int]],action_name_keys: list[int],num_class:int) -> list[int]: 
    """
    convert original predicted actions to classification labels
    """
    # construct a one-hot vector string to int label 
    vec2cls = {"0"*(num_class-1):0}
    for i in range(num_class-1):
        key = "0"*i + "1" + "0"*(num_class-2-i)
        vec2cls[key] = i+1
    # print(vec2cls)
    vec2cls['1'*(num_class-1)] = 0 # do all equal not do 
    # vec2cls = {"00":0,"10":1,"01":2} # tested for class_num = 2 
    num_labels = idm_actions[action_name_keys[0]].size # assert same length: video_num x frame_per_video
    # if not single in first dim, we should perform flattn 
    
    # construct one-hot vector
    idm_action_string = [[str(int(i)) for i in idm_actions[action_name].flatten()] for action_name in action_name_keys]
    try:
        labels = [vec2cls["".join([idm_action_string[j][i] for j in range(num_class-1)])] for i in range(num_labels)]
    except:
        conflicts_num = sum([ i=='1' and j=='1' for i,j in zip(idm_action_string[0],idm_action_string[1])])
        print(f"detect conflict prediction: {conflicts_num}")
        return None 
        
    labels = np.array(labels)
    return labels

def define_exclusive_classification_task(predicted_actions:dict,recorded_actions:dict,calculate_hot_bar = False) -> dict:
    subtasks = {"multi_class":[("back","forward"),# 01,00,10,  
                                ("left","right"),
                                ("sneak","sprint"),
                                ],
                    "binary_class":["use","attack","jump","drop"]
    }
    if calculate_hot_bar:
        subtasks["multi_class"]=[("hotbar.1","hotbar.2","hotbar.3","hotbar.4","hotbar.5","hotbar.6","hotbar.7","hotbar.8","hotbar.9")]
    subtasks_labels = {}
    for class_pair in subtasks["multi_class"]:
        class_num = len(class_pair) + 1 # len = 2 has 00 01 10 
        # convert to strings 
        # convert to classification
        pred_labels = construct_classification_labels(predicted_actions, class_pair, class_num)
        rec_labels = construct_classification_labels(recorded_actions, class_pair, class_num)
        if pred_labels is None or rec_labels is None:
            print(f"detect conflict prediction: {pred_labels} and {rec_labels}")
            continue
        subtasks_labels[class_pair] = {"class_num":class_num,
                                       "pred_labels":pred_labels,
                                       "rec_labels":rec_labels
                                       }
    for binary_task in subtasks["binary_class"]:
        pred_labels =  predicted_actions[binary_task] 
        rec_labels =   recorded_actions[binary_task] 
        subtasks_labels[binary_task] = {"class_num":2,
                                       "pred_labels":pred_labels,
                                       "rec_labels":rec_labels
                                       }
    return subtasks_labels

def classification_metric(pred_labels, rec_labels, class_num):
    ## compute macro and micro score for both tri classification and binary classification
    ## the difference between macro and micro precision and binary precision for binary task is :
    ## the binary precision only compute label with 1 ; but micro and marco compute 0, 1 and then average them 
    ## to align with tri-classification we use average="macro" and average="micro"
    precision_micro = precision_score(rec_labels, pred_labels, average="micro", zero_division=0)
    recall_micro = recall_score(rec_labels, pred_labels, average="micro", zero_division=0)
    f1_micro = f1_score(rec_labels, pred_labels, average="micro", zero_division=0)
    
    precision_macro = precision_score(rec_labels, pred_labels, average="macro", zero_division=0)
    recall_macro = recall_score(rec_labels, pred_labels, average="macro", zero_division=0)
    f1_macro = f1_score(rec_labels, pred_labels, average="macro", zero_division=0)
    return {
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "class_num": class_num
    }

def aggregate_actions(actions:list) -> dict:
    return_dict = {}
    for action in actions:
        for key in action:
            if key not in return_dict:
                return_dict[key] = []
            return_dict[key].append(action[key])
    for key in return_dict:
        return_dict[key] = np.array(return_dict[key]).reshape(-1)
    return return_dict

def idm_prediction(agent, video_path,json_path, infer_demo_num, n_frames):
    th.cuda.empty_cache()
    full_json_data = load_action_jsonl(json_path)
    json_data = full_json_data[infer_demo_num:infer_demo_num+n_frames]
    recorded_actions = [json_action_to_env_action(i)[0] for i in json_data]
    recorded_actions = aggregate_actions(recorded_actions)
    frames = []
    cap = cv2.VideoCapture(video_path)
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"[Error] loading frames in {video_path} returing {_}")
            return None,None
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    frames = np.stack(frames)
    predicted_actions = agent.predict_actions(frames)
    for key in predicted_actions:
        if key == "camera":
            continue
        predicted_actions[key] = np.array(predicted_actions[key]).reshape(-1)
    return predicted_actions,recorded_actions

def camera_loss():
    from lib.actions import CameraQuantizer
    cam_quantizer = CameraQuantizer(
    camera_binsize=2,
    camera_maxval=10,
    mu=10,
    quantization_scheme="mu_law")
    # import pdb;pdb.set_trace()
    cam_pred_token=cam_quantizer.discretize(predicted_actions['camera'])
    cam_gt_token  =cam_quantizer.discretize(np.array(recorded_actions['camera']))
    camera_bin_loss = np.abs(cam_pred_token-cam_gt_token).mean()
    return {
        "camera_bin_loss":camera_bin_loss
    }

if __name__ == "__main__":
    parser = ArgumentParser("Evaluate IDM quality for MC-LVM ")
    parser.add_argument("--weights", type=str, required=True, help="[IDM model config] Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="[IDM model config] Path to the '.model' file to be loaded.")
    parser.add_argument("--jsonl-path", type=str, required=True, help="[Eval Config] Path to .jsonl contains actions.")
    parser.add_argument("--video-path", type=str, required=True, help="[Eval Config] Path to a .mp4 file.")
    parser.add_argument("--infer-demo-num", type=int, default=0, help="[Inference Config] Number of frames to skip before starting evaluation.")
    parser.add_argument("--n-frames", type=int, default=32, help="[Inference Config] Number of frames to generation.")
    parser.add_argument("--output-file", type=str, default="[Eval Config] output/action_loss.jsonl", help="[Eval Config] Path to save the action loss.")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    evaluate_IDM_quality(args.model, args.weights,args.jsonl_path ,args.video_path, args.infer_demo_num,args.n_frames,args.output_file)
