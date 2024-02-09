import numpy as np


def qlearning_dataset(env):
    dataset = env.get_dataset()

    N = dataset["rewards"].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    assert "timeouts" in dataset
    use_timeouts = True

    for i in range(N):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["next_observations"][i].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        is_done = bool(dataset["terminals"][i])
        is_timeout = bool(dataset["timeouts"][i])

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(is_done)

    return {
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
    }


def qlearning_dataset_sarsa(env):
    dataset = env.get_dataset()

    N = dataset["rewards"].shape[0]
    obs_ = []
    next_obs_ = []
    next_acts_ = []
    action_ = []
    reward_ = []
    done_ = []

    assert "timeouts" in dataset
    use_timeouts = True

    for i in range(N - 1):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["next_observations"][i].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        next_act = dataset["actions"][i + 1].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        is_done = bool(dataset["terminals"][i])
        is_timeout = bool(dataset["timeouts"][i])
        if not is_timeout and not is_done:
            assert (new_obs == dataset["observations"][i + 1].astype(np.float32)).all()

        if is_timeout and not is_done:
            continue

        obs_.append(obs)
        next_obs_.append(new_obs)
        next_acts_.append(next_act)
        action_.append(action)
        reward_.append(reward)
        done_.append(is_done)

    return {
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "next_actions": np.array(next_acts_),
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
    }
