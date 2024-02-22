from Common.utils import *
from utils.reporter import get_reporter


class TrainEnvPool:
    def __init__(self, **config):
        self.config = config
        self.env_name = self.config["env_name"]
        self.max_episode_steps = self.config["max_steps_per_episode"]
        self.state_shape = self.config["state_shape"]
        self.env = make_atari(
            self.env_name,
            self.config["n_workers"],
            self.max_episode_steps,
            seed=config["seed"],
        )
        self.reset()

    def reset(self):
        state, info = self.env.reset()
        return state, {"elapsed_step": info["elapsed_step"]}

    def step(self, actions):
        next_state, r, term, trunc, info = self.env.step(actions)
        return (
            next_state,
            r,
            term,
            trunc,
            {"elapsed_step": info["elapsed_step"]},
        )

    def reset_dead_envs(self, next_state, r, term, trunc, info):
        stop = np.logical_or(term.astype(bool), trunc.astype(bool))

        if not np.any(stop):
            return next_state, info

        dead_idx = np.argwhere(stop).flatten()
        new_states, new_info = self.env.reset(dead_idx)

        assert new_states.shape == (dead_idx.shape[0], 4, 84, 84)

        next_state[dead_idx] = new_states

        rlt_info = {}
        rlt_info["elapsed_step"] = info["elapsed_step"]
        rlt_info["elapsed_step"][dead_idx] = new_info["elapsed_step"]
        assert np.all(rlt_info["elapsed_step"][dead_idx] == 0)

        return next_state, rlt_info


class EvalEnvPool:
    def __init__(self, **config):
        self.config = config
        self.env_name = self.config["env_name"]
        self.max_episode_steps = self.config["max_steps_per_episode"]
        self.state_shape = self.config["state_shape"]
        self.env = make_atari(
            self.env_name,
            self.config["eval_n_workers"],
            self.max_episode_steps,
            seed=config["seed"] * 10,
        )
        self.env_num = self.config["eval_n_workers"]
        self.reset()

    @property
    def nums(self):
        return self.env_num

    def reset(self):
        state, info = self.env.reset()
        self.env_is_alive = np.ones((self.nums,), dtype=np.float32)
        return state, {"elapsed_step": info["elapsed_step"]}

    def step(self, actions):
        next_state, r, term, trunc, info = self.env.step(actions)
        return (
            next_state,
            r * self.env_is_alive,
            term,
            trunc,
            {"elapsed_step": info["elapsed_step"]},
        )

    def mark_dead_envs(self, next_state, r, term, trunc, info):
        stop = np.logical_or(term.astype(bool), trunc.astype(bool))

        if not np.any(stop):
            return

        dead_idx = np.argwhere(stop).flatten()
        self.env_is_alive[dead_idx] = 0.0

    def is_all_dead(self):
        return np.all(self.env_is_alive == 0.0)


@torch.no_grad()
def eval_in_envs(get_action, eval_envs: EvalEnvPool, counter: int):
    rets = np.zeros((eval_envs.nums,), dtype=np.float32)
    s, i = eval_envs.reset()

    while not eval_envs.is_all_dead():

        acts = get_action(s)

        assert acts.shape == (eval_envs.nums,)

        s, r, term, trunc, i = eval_envs.step(acts)

        assert rets.shape == r.shape
        rets += r

        eval_envs.mark_dead_envs(s, r, term, trunc, i)

    get_reporter().add_distributions(
        {
            # "Return Mean": (np.mean(rets), counter),
            # "Return Std": (np.std(rets), counter),
            "returns": (torch.as_tensor(rets, dtype=torch.float32), counter)
        },
        "eval",
    )
    get_reporter().add_scalars(
        {
            "Return Mean": (np.mean(rets), counter),
            "Return Std": (np.std(rets), counter),
        },
        "eval",
    )
    return rets, dict(return_mean=np.mean(rets), return_std=np.std(rets))
