from Common.utils import *


class EnvPoolWorker:
    def __init__(self, id, **config):
        self.id = id
        self.config = config
        self.env_name = self.config["env_name"]
        self.max_episode_steps = self.config["max_steps_per_episode"]
        self.state_shape = self.config["state_shape"]
        self.env = make_atari(
            self.env_name,
            self.config["n_workers"],
            self.max_episode_steps,
            seed=config["seed"] + id + 1,
        )
        self.reset()

    def __str__(self):
        return str(self.id)

    def reset(self):
        state, info = self.env.reset()
        return state, {"elapsed_step": info["elapsed_step"]}

    def step(self, actions):
        next_state, r, term, trunc, info = self.env.step(actions)
        return (
            next_state,
            np.sign(r),
            term,
            trunc,
            {"elapsed_step": info["elapsed_step"]},
        )

    def reset_dead_envs(self, next_state, r, term, trunc, info):
        stop = np.logical_or(term.astype(bool), trunc.astype(bool))

        if not np.any(stop):
            return next_state, info

        dead_idx = np.argwhere(stop).flatten()
        assert np.all(r[dead_idx] == 0.0)
        new_states, new_info = self.env.reset(dead_idx)

        assert new_states.shape == (dead_idx.shape[0], 4, 84, 84)

        next_state[dead_idx] = new_states

        rlt_info = {}
        rlt_info["elapsed_step"] = info["elapsed_step"]
        rlt_info["elapsed_step"][dead_idx] = new_info["elapsed_step"]
        assert np.all(rlt_info["elapsed_step"][dead_idx] == 0)

        return next_state, rlt_info
