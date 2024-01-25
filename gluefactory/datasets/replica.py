import argparse
import logging
import shutil
import tarfile
from collections.abc import Iterable
from pathlib import Path

import re
import h5py
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch
import json
from omegaconf import OmegaConf

from ..settings import DATA_PATH
from ..utils.image import ImagePreprocessor, load_image
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class Replica(BaseDataset):
    default_conf = {
        # path
        "data_dir": "replica",
        "image_dir": "col",
        "scenes_names": ["office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2",],
        "cam_info_dir": "cam_params.json",
        # train
        "train": True,
        "train_pairs": "pairs.txt",
        "train_scenes": ["office0", "office1", "office2", "office3", "office4", "room0", "room1", ],
        # validation
        "val": True,
        "val_pairs": "pairs.txt",
        "val_scenes": ["room2", ],
        # image
        "preprocessing": None,

    }

    def _init(self, conf):
        if not (DATA_PATH / conf.data_dir).exists():
            logger.info("Downloading the MegaDepth dataset.")
            self.download()

    def download(self):
        data_dir = DATA_PATH / self.conf.data_dir
        tmp_dir = data_dir.parent / "replica_tmp"
        if tmp_dir.exists():  # The previous download failed.
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(exist_ok=True, parents=True)
        url_base = "https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip"
        tar_path = tmp_dir / "Replica.zip"
        torch.hub.download_url_to_file(url_base, tmp_dir)
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=tmp_dir)
        tar_path.unlink()
        shutil.move(tmp_dir / "Replica", data_dir)
        
    def get_dataset(self, split):
        return _Dataset(self.conf, split)
    

class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split):
        self.conf = conf
        self.split = split
        self.data_dir = DATA_PATH / conf.data_dir
        self.scenes_names = conf[f"{self.split}_scenes"]
        self.cam_info_dir = conf.cam_info_dir
        self.pairs = self._get_pairs()
        self.image_preprocessor = ImagePreprocessor(**conf.preprocessing) if conf.preprocessing else None

    def _get_pairs(self):
        pairs = []
        for scene_name in self.scenes_names:
            pairs_path = self.data_dir  / self.conf[f"{self.split}_pairs"] # replica/pairs.txt
            with open(pairs_path) as f:
                pairs += [[f"{scene_name}/{i}.jpg" for i in line.split()] for line in f.read().splitlines()]
        return pairs
    
    def __getitem__(self, idx):
        return self.getitem(idx)
        
    def get_intrinsics(self, scene_name):
        # json file
        cam_info_path = self.data_dir / self.cam_info_dir
        with open(cam_info_path) as f:
            cam_info = json.load(f)
        fx = cam_info["camera"]["fx"]
        fy = cam_info["camera"]["fy"]
        cx = cam_info["camera"]["cx"]
        cy = cam_info["camera"]["cy"]
        intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return intrinsics
    

    def getitem(self, idx):
        pair = self.pairs[idx]
        scene_name = pair[0].split("/")[0]
        intrinsics = self.get_intrinsics(scene_name)

        data0 = self._read_view(scene_name, pair[0])
        data1 = self._read_view(scene_name, pair[1])

        if self.image_preprocessor:
            image0, image1 = self.image_preprocessor(image0, image1)
        data = {
            "view0": data0,
            "view1": data1,
            "T_0to1": data1["pose"] @ torch.inverse(data0["pose"]),
            "T_1to0": data0["pose"] @ torch.inverse(data1["pose"]),
        }
        data["view0"]["intrinsics"] = intrinsics
        data["view1"]["intrinsics"] = intrinsics

        return data
    
    def _read_view(self, scene_name, view_name):
        image_path = self.data_dir / view_name.split("/")[0] / self.conf.image_dir / view_name.split("/")[1]
        image = load_image(image_path)
        traj_path = self.data_dir / f"{scene_name}/traj.txt"
        view = view_name.split("/")[1]
        num = int(re.findall(r"\d+", view)[0])
        
        with open(traj_path) as f:
            traj = f.read().splitlines()
        pose = traj[num].split()
        pose = np.array(pose, dtype=np.float32)
        pose = torch.tensor(pose.reshape(4, 4))
        data = {
            "image": image,
            "intrinsics": None,
            "pose": pose,
        }
        return data

    def __len__(self):
        return len(self.pairs)
    

