from Common.runner import TrainEnvPool, EvalEnvPool, eval_in_envs
from Common.play import Play
from Common.config import get_params
from Common.logger import Logger
from torch.multiprocessing import Process, Pipe
import numpy as np
from Brain.brain import Brain
import gym
from utils.progress import tqdm
from utils.seed import seed_all
from utils.reporter import init_reporter, get_reporter
import json
import envpool


def run_workers(worker, conn):
    worker.step(conn)


if __name__ == "__main__":
    config = get_params()
    init_reporter(config["name"], json.dumps(config, indent=4, sort_keys=True))
    seed_all(config["seed"])

    test_env = envpool.make("MontezumaRevenge-v5", env_type="gym", num_envs=1)
    config.update({"n_actions": test_env.action_space.n})
    test_env.close()
    del test_env

    config.update(
        {
            "batch_size": (config["rollout_length"] * config["n_workers"])
            // config["n_mini_batch"]
        }
    )
    config.update({"predictor_proportion": 32 / config["n_workers"]})

    brain = Brain(**config)
    brain.set_to_train_mode()
    logger = Logger(brain, **config)

    if not config["do_test"]:
        if not config["train_from_scratch"]:
            checkpoint = logger.load_weights()
            brain.set_from_checkpoint(checkpoint)
            running_ext_reward = checkpoint["running_reward"]
            init_iteration = checkpoint["iteration"]
            episode = checkpoint["episode"]
            visited_rooms = checkpoint["visited_rooms"]
            logger.running_ext_reward = running_ext_reward
            logger.episode = episode
            logger.visited_rooms = visited_rooms
        else:
            init_iteration = 0
            running_ext_reward = 0
            episode = 0
            visited_rooms = set([1])

        envs = TrainEnvPool(**config)
        eval_envs = EvalEnvPool(**config)

        if config["train_from_scratch"]:
            print("---Pre_normalization started.---")
            envs.reset()
            states = []
            total_pre_normalization_steps = (
                config["rollout_length"] * config["pre_normalization_steps"]
            )
            actions = np.random.randint(
                0,
                config["n_actions"],
                size=(total_pre_normalization_steps, config["n_workers"]),
            )
            for t in tqdm(
                range(total_pre_normalization_steps), desc="pre_normalization"
            ):

                s_, *_extra = envs.step(actions[t])
                assert s_.shape == (config["n_workers"], 4, 84, 84), s_.shape

                states.append(s_[:, -1, ...].reshape(config["n_workers"], 1, 84, 84))

                if (len(states) * config["n_workers"]) % (
                    config["n_workers"] * config["rollout_length"]
                ) == 0:
                    brain.state_rms.update(np.concatenate(states, axis=0))
                    states = []

                envs.reset_dead_envs(s_, *_extra)
            print("---Pre_normalization is done.---")

        # prepare to start training iterations
        rollout_base_shape = config["n_workers"], config["rollout_length"]
        init_states = np.zeros(
            rollout_base_shape + config["state_shape"], dtype=np.uint8
        )
        assert init_states.shape == (
            config["n_workers"],
            config["rollout_length"],
            *config["state_shape"],
        )
        init_actions = np.zeros(rollout_base_shape, dtype=np.uint8)
        init_action_probs = np.zeros(rollout_base_shape + (config["n_actions"],))
        init_int_rewards = np.zeros(rollout_base_shape)
        init_ext_rewards = np.zeros(rollout_base_shape)
        init_dones = np.zeros(rollout_base_shape, dtype=bool)
        init_int_values = np.zeros(rollout_base_shape)
        init_ext_values = np.zeros(rollout_base_shape)
        init_log_probs = np.zeros(rollout_base_shape)
        init_next_states = np.zeros(
            (rollout_base_shape[0],) + config["state_shape"], dtype=np.uint8
        )
        init_next_obs = np.zeros(
            rollout_base_shape + config["obs_shape"], dtype=np.uint8
        )

        logger.on()
        episode_ext_reward = 0
        concatenate = np.concatenate

        _states, _infos = envs.reset()

        brain.set_to_eval_mode()
        eval_in_envs(
            lambda s: brain.get_actions_and_values(s, batch=True)[0], eval_envs, 0
        )
        brain.set_to_train_mode()

        assert config["total_rollouts_per_env"] % config["eval_times"] == 0
        for iteration in tqdm(
            range(init_iteration + 1, config["total_rollouts_per_env"] + 1),
            desc="total iter: ",
        ):
            total_states = init_states
            total_actions = init_actions
            total_action_probs = init_action_probs
            total_int_rewards = init_int_rewards
            total_ext_rewards = init_ext_rewards
            total_dones = init_dones
            total_int_values = init_int_values
            total_ext_values = init_ext_values
            total_log_probs = init_log_probs
            next_states = init_next_states
            total_next_obs = init_next_obs

            for t in range(config["rollout_length"]):
                # for worker_id, parent in enumerate(parents):
                #     total_states[worker_id, t] = parent.recv()
                assert _states.shape == (config["n_workers"], 4, 84, 84)
                total_states[:, t] = _states

                (
                    total_actions[:, t],
                    total_int_values[:, t],
                    total_ext_values[:, t],
                    total_log_probs[:, t],
                    total_action_probs[:, t],
                ) = brain.get_actions_and_values(total_states[:, t], batch=True)

                s_, original_reward, _terms, _truncs, info = envs.step(
                    total_actions[:, t]
                )
                assert s_.shape == (config["n_workers"], 4, 84, 84)
                r = np.sign(original_reward)

                total_ext_rewards[:, t] = r
                total_dones[:, t] = _terms
                next_states[:] = s_
                total_next_obs[:, t] = s_[:, [-1], ...]
                episode_ext_reward += original_reward[0]

                # 第0号环境done
                if _terms[0] or _truncs[0]:
                    episode += 1
                    # if "episode" in infos[0]:
                    # visited_rooms = infos[0]["episode"]["visited_room"]
                    visited_rooms = []
                    logger.log_episode(episode, episode_ext_reward, visited_rooms)
                    episode_ext_reward = 0

                s_, info = envs.reset_dead_envs(s_, r, _terms, _truncs, info)

                _states = s_

            total_next_obs = concatenate(total_next_obs)
            assert total_next_obs.shape == (
                config["n_workers"] * config["rollout_length"],
                1,
                84,
                84,
            )
            total_int_rewards = brain.calculate_int_rewards(total_next_obs)
            _, next_int_values, next_ext_values, *_ = brain.get_actions_and_values(
                next_states, batch=True
            )

            total_int_rewards = brain.normalize_int_rewards(total_int_rewards)

            training_logs = brain.train(
                states=concatenate(total_states),
                actions=concatenate(total_actions),
                int_rewards=total_int_rewards,
                ext_rewards=total_ext_rewards,
                dones=total_dones,
                int_values=total_int_values,
                ext_values=total_ext_values,
                log_probs=concatenate(total_log_probs),
                next_int_values=next_int_values,
                next_ext_values=next_ext_values,
                total_next_obs=total_next_obs,
            )

            logger.log_iteration(
                iteration,
                training_logs,
                total_int_rewards[0].mean(),
                total_action_probs[0].max(-1).mean(),
            )

            if iteration % int(config["total_rollouts_per_env"] / config["eval_times"]) == 0:

                brain.set_to_eval_mode()
                eval_in_envs(
                    lambda s: brain.get_actions_and_values(s, batch=True)[0],
                    eval_envs,
                    iteration,
                )
                brain.set_to_train_mode()

    else:
        checkpoint = logger.load_weights()
        play = Play(config["env_name"], brain, checkpoint)
        play.evaluate(config["seed"])
