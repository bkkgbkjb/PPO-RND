from typing import Dict, Any, Optional, Tuple
from os import path
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from utils.time import get_current_datetime_str


class Reporter:
    def __init__(self, writer, counter):
        self._writer = writer
        self._times_counter = counter

    def add_scalars(self, info: Dict[str, Any], prefix: str):
        for _k, v in info.items():
            k = f"{prefix}/{_k}"

            if isinstance(v, tuple):
                assert isinstance(v[1], int)
                assert k not in self._times_counter or self._times_counter[k] < v[1]
                val = v[0]
                self._writer.add_scalar(k, val, v[1])
                self._times_counter[k] = v[1]
            else:
                if not k in self._times_counter:
                    self._times_counter[k] = 0
                val = v
                self._writer.add_scalar(k, val, self._times_counter[k])
                self._times_counter[k] += 1

    def add_distributions(self, info: Dict[str, Any], prefix: str):
        print("------------Important Distributions Data-------------")
        for _k, v in info.items():
            k = f"{prefix}/{_k}"

            if isinstance(v, tuple):
                assert isinstance(v[1], int)
                assert k not in self._times_counter or self._times_counter[k] < v[1]
                val = v[0]
                assert isinstance(val, torch.Tensor)
                print(f"{k}: {val.tolist()} @ {v[1]}")
                print(
                    f"{k}: mean: {val.mean().item()}, std: {val.std().item()} @ {v[1]}"
                )
                self._times_counter[k] = v[1]
            else:
                if not k in self._times_counter:
                    self._times_counter[k] = 0
                val = v
                assert isinstance(val, torch.Tensor)
                print(f"{k}: {val.tolist()} @ {self._times_counter[k]}")
                print(
                    f"{k}: mean: {val.mean().item()}, std: {val.std().item()} @ {self._times_counter[k]}"
                )
                self._times_counter[k] += 1
        print("---------------End--------------------", flush=True)

    def add_videos(self, info: Dict[str, Tuple[np.ndarray, int]], prefix: str):
        for _k, (video, step) in info.items():
            k = f"{prefix}/{_k}"

            self._writer.add_video(k, video, step)

    def add_params(self, params: Dict[str, Any]):
        for k, v in params.items():
            self._writer.add_text(k, str(v))

    def add_text(self, tag: str, text: str):
        self._writer.add_text(tag, text)


reporter: Optional[Reporter] = None
reporter_dir: Optional[str] = None


def get_reporter():
    assert reporter is not None
    return reporter


def get_reporter_dir():
    assert reporter_dir is not None
    return reporter_dir


def init_reporter(name: str, info: Optional[str] = None):
    global reporter, reporter_dir
    assert reporter is None and reporter_dir is None

    writer = SummaryWriter(
        log_dir=path.join("tblogs", get_current_datetime_str() + "_" + name)
    )
    times_counter: Dict[str, int] = dict()

    if info is not None:
        assert writer.log_dir is not None
        with open(path.join(writer.log_dir, "info.txt"), "w") as f:
            f.write(info)

    reporter = Reporter(writer, times_counter)
    reporter_dir = writer.get_logdir()
    return
