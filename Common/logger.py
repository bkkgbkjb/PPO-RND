import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import datetime
import glob
from collections import deque
from utils.reporter import get_reporter
from utils.time import get_current_datetime_str


class Logger:
    def __init__(self, brain, **config):
        self.config = config
        self.brain = brain
        self.log_dir = get_current_datetime_str()
        self.start_time = 0
        self.duration = 0
        self.episode = 0
        self.episode_ext_reward = 0
        self.running_ext_reward = 0
        self.running_int_reward = 0
        self.running_act_prob = 0
        self.running_training_logs = 0
        self.visited_rooms = set([1])
        self.max_episode_reward = -np.inf
        self.moving_avg_window = 10
        self.moving_weights = np.repeat(1.0, self.moving_avg_window) / self.moving_avg_window
        self.last_10_ep_rewards = deque(maxlen=10)
        self.running_last_10_ext_r = 0  # It is not correct but does not matter.

        if not self.config["do_test"] and self.config["train_from_scratch"]:
            self.create_wights_folder()
            self.log_params()

        self.exp_avg = lambda x, y: 0.9 * x + 0.1 * y if (y != 0).all() else y

    def create_wights_folder(self):
        if not os.path.exists("Models"):
            os.mkdir("Models")
        os.mkdir("Models/" + self.log_dir)

    def log_params(self):
        # with SummaryWriter("Logs/" + self.log_dir) as writer:
        # for k, v in self.config.items():
        get_reporter().add_params(self.config)

    def on(self):
        self.start_time = time.time()

    def off(self):
        self.duration = time.time() - self.start_time

    def log_iteration(self, *args):
        iteration, training_logs, int_reward, action_prob = args

        self.running_act_prob = self.exp_avg(self.running_act_prob, action_prob)
        self.running_int_reward = self.exp_avg(self.running_int_reward, int_reward)
        self.running_training_logs = self.exp_avg(self.running_training_logs, np.array(training_logs))

        if iteration % (self.config["interval"] // 3) == 0:
            self.save_params(self.episode, iteration)

        get_reporter().add_scalars(
            {
                # "Episode Ext Reward": (self.episode_ext_reward, self.episode),
                # "Running Episode Ext Reward": (self.running_ext_reward, self.episode),
                # "Visited rooms": (len(list(self.visited_rooms)), self.episode),
                # "Running last 10 Ext Reward": (
                #     self.running_last_10_ext_r,
                #     self.episode,
                # ),
                # "Max Episode Ext Reward": (self.max_episode_reward, self.episode),
                "Running Action Probability": (self.running_act_prob, iteration),
                "Running Intrinsic Reward": (self.running_int_reward, iteration),
                "Running PG Loss": (self.running_training_logs[0], iteration),
                "Running Ext Value Loss": (self.running_training_logs[1], iteration),
                "Running Int Value Loss": (self.running_training_logs[2], iteration),
                "Running RND Loss": (self.running_training_logs[3], iteration),
                "Running Entropy": (self.running_training_logs[4], iteration),
                "Running Intrinsic Explained variance": (
                    self.running_training_logs[5],
                    iteration,
                ),
                "Running Extrinsic Explained variance": (
                    self.running_training_logs[6],
                    iteration,
                ),
            },
            "train",
        )

        self.off()
        if iteration % self.config["interval"] == 0:
            print("Iter:{}| "
                  "EP:{}| "
                  "EP_Reward:{}| "
                  "EP_Running_Reward:{:.3f}| "
                  "Visited_rooms:{}| "
                  "Iter_Duration:{:.3f}| "
                  "Time:{} "
                  .format(iteration,
                          self.episode,
                          self.episode_ext_reward,
                          self.running_ext_reward,
                          self.visited_rooms,
                          self.duration,
                          datetime.datetime.now().strftime("%H:%M:%S"),
                          ))
        self.on()

    def log_episode(self, *args):
        self.episode, self.episode_ext_reward, self.visited_rooms = args

        self.max_episode_reward = max(self.max_episode_reward, self.episode_ext_reward)

        self.running_ext_reward = self.exp_avg(self.running_ext_reward, self.episode_ext_reward)
        get_reporter().add_scalars(
            {
                "Episode Ext Reward": (self.episode_ext_reward, self.episode),
                "Running Episode Ext Reward": (self.running_ext_reward, self.episode),
                "Visited rooms": (len(list(self.visited_rooms)), self.episode),
                "Running last 10 Ext Reward": (
                    self.running_last_10_ext_r,
                    self.episode,
                ),
                "Max Episode Ext Reward": (self.max_episode_reward, self.episode),
            },
            "train",
        )

        self.last_10_ep_rewards.append(self.episode_ext_reward)
        if len(self.last_10_ep_rewards) == self.moving_avg_window:
            self.running_last_10_ext_r = np.convolve(self.last_10_ep_rewards, self.moving_weights, 'valid')

    def save_params(self, episode, iteration):
        torch.save({"current_policy_state_dict": self.brain.current_policy.state_dict(),
                    "predictor_model_state_dict": self.brain.predictor_model.state_dict(),
                    "target_model_state_dict": self.brain.target_model.state_dict(),
                    "optimizer_state_dict": self.brain.optimizer.state_dict(),
                    "state_rms_mean": self.brain.state_rms.mean,
                    "state_rms_var": self.brain.state_rms.var,
                    "state_rms_count": self.brain.state_rms.count,
                    "int_reward_rms_mean": self.brain.int_reward_rms.mean,
                    "int_reward_rms_var": self.brain.int_reward_rms.var,
                    "int_reward_rms_count": self.brain.int_reward_rms.count,
                    "iteration": iteration,
                    "episode": episode,
                    "running_reward": self.running_ext_reward,
                    "visited_rooms": self.visited_rooms
                    },
                   "Models/" + self.log_dir + "/params.pth")

    def load_weights(self):
        model_dir = glob.glob("Models/*")
        model_dir.sort()
        checkpoint = torch.load(model_dir[-1] + "/params.pth")
        self.log_dir = model_dir[-1].split(os.sep)[-1]
        return checkpoint
