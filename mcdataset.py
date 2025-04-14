#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import attr
import collections
import numpy as np
from typing import Union, Dict
import torch
from utils import print0


# https://github.com/openai/Video-Pre-Training/blob/aed46b90e8db2332801feabd8be2de01f92c0ad2/run_inverse_dynamics_model.py#L17
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

# https://github.com/openai/Video-Pre-Training/blob/aed46b90e8db2332801feabd8be2de01f92c0ad2/run_inverse_dynamics_model.py#L41
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
    "camera": np.array([0, 0]),  # [y, x]
    "attack": 0,
    "use": 0,
    "pickItem": 0,
}

OASIS_ACTION_KEYS = [
    "inventory",
    "ESC",
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
    "forward",
    "back",
    "left",
    "right",
    "cameraX",
    "cameraY",
    "jump",
    "sneak",
    "sprint",
    "swapHands",
    "attack",
    "use",
    "pickItem",
    "drop",
]


# Matches a number in the MineRL Java code regarding sensitivity
# This is for mapping from recorded sensitivity to the one used in the model
CAMERA_SCALER = 360.0 / 2400.0


# https://github.com/openai/Video-Pre-Training/blob/main/lib/actions.py#L8 with some modifications
class Buttons:
    # 14 in total without hotbar and camera
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
    INVENTORY = "inventory"
    # added by Yang
    ESC = "ESC"
    SWAPHANDS = "swapHands"
    PICKITEM = "pickItem"

    ALL = [
        USE,
        ATTACK,

        
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

        INVENTORY,
        ESC,
    ] + [f"hotbar.{i}" for i in range(1, 10)]


class QuantizationScheme:
    LINEAR = "linear"
    MU_LAW = "mu_law"


# https://github.com/openai/Video-Pre-Training/blob/main/lib/actions.py#L49
@attr.s(auto_attribs=True)
class CameraQuantizer:
    """
    A camera quantizer that discretizes and undiscretizes a continuous camera input with y (pitch) and x (yaw) components.

    Parameters:
    - camera_binsize: The size of the bins used for quantization. In case of mu-law quantization, it corresponds to the average binsize.
    - camera_maxval: The maximum value of the camera action.
    - quantization_scheme: The quantization scheme to use. Currently, two quantization schemes are supported:
    - Linear quantization (default): Camera actions are split uniformly into discrete bins
    - Mu-law quantization: Transforms the camera action using mu-law encoding (https://en.wikipedia.org/wiki/%CE%9C-law_algorithm)
    followed by the same quantization scheme used by the linear scheme.
    - mu: Mu is the parameter that defines the curvature of the mu-law encoding. Higher values of
    mu will result in a sharper transition near zero. Below are some reference values listed
    for choosing mu given a constant maxval and a desired max_precision value.
    maxval = 10 | max_precision = 0.5  | μ ≈ 2.93826
    maxval = 10 | max_precision = 0.4  | μ ≈ 4.80939
    maxval = 10 | max_precision = 0.25 | μ ≈ 11.4887
    maxval = 20 | max_precision = 0.5  | μ ≈ 2.7
    maxval = 20 | max_precision = 0.4  | μ ≈ 4.39768
    maxval = 20 | max_precision = 0.25 | μ ≈ 10.3194
    maxval = 40 | max_precision = 0.5  | μ ≈ 2.60780
    maxval = 40 | max_precision = 0.4  | μ ≈ 4.21554
    maxval = 40 | max_precision = 0.25 | μ ≈ 9.81152
    """

    camera_maxval: int
    camera_binsize: int
    quantization_scheme: str = attr.ib(
        default=QuantizationScheme.LINEAR,
        validator=attr.validators.in_([QuantizationScheme.LINEAR, QuantizationScheme.MU_LAW]),
    )
    mu: float = attr.ib(default=5)

    def discretize(self, xy):
        xy = np.clip(xy, -self.camera_maxval, self.camera_maxval)

        if self.quantization_scheme == QuantizationScheme.MU_LAW:
            xy = xy / self.camera_maxval
            v_encode = np.sign(xy) * (np.log(1.0 + self.mu * np.abs(xy)) / np.log(1.0 + self.mu))
            v_encode *= self.camera_maxval
            xy = v_encode

        # Quantize using linear scheme
        return np.round((xy + self.camera_maxval) / self.camera_binsize).astype(np.int64)

    def undiscretize(self, xy):
        xy = xy * self.camera_binsize - self.camera_maxval

        if self.quantization_scheme == QuantizationScheme.MU_LAW:
            xy = xy / self.camera_maxval
            v_decode = np.sign(xy) * (1.0 / self.mu) * ((1.0 + self.mu) ** np.abs(xy) - 1.0)
            v_decode *= self.camera_maxval
            xy = v_decode
        return xy


class MCDataset(torch.utils.data.Dataset):
    """
    Dataset for Minecraft.
    """
    def __init__(self,
                 action_length: int = 11,  # including bos and eos
                 camera_binsize: int = 9,  # 2 in vpt
                 camera_maxval: int = 90,  # 10 in vpt
                 camera_mu: float = 11.4887,  # 10 in vpt
                 quantization_scheme: str = "mu_law",
    ):
        self.action_length = action_length
        self.camera_quantizer = CameraQuantizer(
            camera_binsize=camera_binsize,
            camera_maxval=camera_maxval,
            mu=camera_mu,
            quantization_scheme=quantization_scheme,
        )

    def json_action_to_env_action(self, json_action):
        """
        https://github.com/openai/Video-Pre-Training/blob/aed46b90e8db2332801feabd8be2de01f92c0ad2/run_inverse_dynamics_model.py#L80
        Converts a json action into a MineRL action.
        Returns (minerl_action, is_null_action)
        """
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

        # added by Yang
        # if two confictory actions are pressed, remove them
        if env_action["forward"] == 1 and env_action["back"] == 1:
            env_action["forward"] = 0
            env_action["back"] = 0
        if env_action["left"] == 1 and env_action["right"] == 1:
            env_action["left"] = 0
            env_action["right"] = 0 
        if env_action["jump"] == 1 and env_action["sneak"] == 1:
            env_action["jump"] = 0
            env_action["sneak"] = 0
        if env_action["sprint"] == 1 and env_action["sneak"] == 1:
            env_action["sprint"] = 0
            env_action["sneak"] = 0
        if env_action["attack"] == 1 and env_action["use"] == 1:
            env_action["attack"] = 0
            env_action["use"] = 0

        # remove inventory and ESC action
        if env_action["inventory"] == 1:
            is_null_action = True
        if env_action["ESC"] == 1:
            is_null_action = True

        return env_action, is_null_action

    def make_action_vocab(self,
                          num_cam_bins: int = 21,
                          action_vocab_offset: int = 0,
                          verbose: bool = False):
        action_vocab = collections.OrderedDict()
        # 14 actions and hotbar.1-9
        for i, action in enumerate(Buttons.ALL):
            action_vocab[action] = i
        # camera 0 
        for i in range(num_cam_bins):
            action_vocab[f"cam_0_{i}"] = len(Buttons.ALL) + i
        # camera 1
        for i in range(num_cam_bins):
            action_vocab[f"cam_1_{i}"] = len(Buttons.ALL) + num_cam_bins + i
        # bos, null, eos
        action_vocab["<act_bos>"] = len(Buttons.ALL) + 2 * num_cam_bins
        action_vocab["<null_act>"] = len(Buttons.ALL) + 2 * num_cam_bins + 1
        action_vocab["<act_eos>"] = len(Buttons.ALL) + 2 * num_cam_bins + 2

        if action_vocab_offset > 0:
            action_vocab = {k: v + action_vocab_offset for k, v in action_vocab.items()}

        if verbose:
            print0(f"[bold yellow]\[MCDataset][/bold yellow] Action Vocab: {action_vocab}")

        self.action_vocab = action_vocab
        # return action_vocab

    def _handle_conflict_action_index(self,
                                action_dict: Dict[str, Union[int, np.ndarray]],
                                key1: str,
                                key2: str,
                                null_key: str,
                                verbose: bool = False):
        if action_dict[key1] == 1 and action_dict[key2] == 1:
            if verbose:
                print0(f"[bold yellow]\[MCDataset][/bold yellow] {key1} and {key2} are both pressed")
            return self.action_vocab[null_key]
        elif action_dict[key1] == 1:
            return self.action_vocab[key1]
        elif action_dict[key2] == 1:
            return self.action_vocab[key2]
        else:
            return self.action_vocab[null_key]

    def get_action_index_from_actiondict(self,
                                         action_dict: Dict[str, Union[int, np.ndarray]],
                                         action_vocab_offset: int = 0,
                                         verbose: bool = False):

        if not hasattr(self, "action_vocab"):
            self.make_action_vocab(action_vocab_offset=action_vocab_offset, verbose=verbose)

        # action_list = [boa, camy, camx, hotbar, fore_back, left_right, sprint_sneak, use_attack, jump, drop_pick, eoa]
        # 11 actions
        action_list = [self.action_vocab["<null_act>"]] * self.action_length
        # 0 & 10
        action_list[0] = self.action_vocab["<act_bos>"]
        action_list[-1] = self.action_vocab["<act_eos>"]

        camera_action = action_dict["camera"]
        assert len(camera_action) == 2, f"[MCDataset] camera_action length is not 2: {camera_action}"
        # camera_action should be numpy array
        if not isinstance(camera_action, np.ndarray):
            camera_action = np.array(camera_action)
        camera_action = self.camera_quantizer.discretize(camera_action)
        # 1 & 2
        action_list[1] = self.action_vocab[f"cam_0_{camera_action[0]}"]
        action_list[2] = self.action_vocab[f"cam_1_{camera_action[1]}"]

        # 3
        for i in range(1, 10):
            if f"hotbar.{i}" in action_dict and action_dict[f"hotbar.{i}"] == 1:
                action_list[3] = self.action_vocab[f"hotbar.{i}"]
                break

        # 4 forward/back
        action_list[4] = self._handle_conflict_action_index(action_dict, "forward", "back", "<null_act>", verbose=verbose)
        # 5 left/right
        action_list[5] = self._handle_conflict_action_index(action_dict, "left", "right", "<null_act>", verbose=verbose)
        # 6 sprint/sneak
        action_list[6] = self._handle_conflict_action_index(action_dict, "sprint", "sneak", "<null_act>", verbose=verbose)
        # 7 use/attack
        action_list[7] = self._handle_conflict_action_index(action_dict, "use", "attack", "<null_act>", verbose=verbose)
        # 8 jump
        action_list[8] = self.action_vocab["jump"] if action_dict["jump"] == 1 else self.action_vocab["<null_act>"]
        # 9 drop/pick
        action_list[9] = self._handle_conflict_action_index(action_dict, "drop", "pickItem", "<null_act>", verbose=verbose)

        if verbose:
            print0(f"[bold yellow]\[MCDataset][/bold yellow] Action List: {action_list}")

        return action_list

    def read_jsonl(self, jsonl_path: str):
        assert os.path.isfile(jsonl_path), f"[MCDataset] {jsonl_path} does not exist"
        # read jsonl
        # https://github.com/openai/Video-Pre-Training/blob/main/data_loader.py#L76
        try:
            with open(jsonl_path) as json_file:
                json_lines = json_file.readlines()
                json_data = "[" + ",".join(json_lines) + "]"
                json_data = json.loads(json_data)
        except Exception as e:
            print0(f"[bold yellow]\[MCDataset][/bold yellow] {jsonl_path} cannot be read: {e}")
            return None
        return json_data