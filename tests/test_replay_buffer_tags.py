import numpy as np
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from maddpg.trainer.replay_buffer import ReplayBuffer


def test_tagged_sampling():
    buffer = ReplayBuffer(size=50000)

    for episode_id in range(1, 4001):
        tag = "malicious" if 2000 <= episode_id <= 3000 else "normal"
        for step in range(2):
            obs = np.array([episode_id, step], dtype=np.float32)
            action = np.array([step], dtype=np.float32)
            reward = float(step)
            next_obs = np.array([episode_id, step + 1], dtype=np.float32)
            done = float(step == 1)
            buffer.add(
                obs,
                action,
                reward,
                next_obs,
                done,
                episode_id=episode_id,
                tag=tag,
            )

    malicious_episode_count = 3000 - 2000 + 1
    expected_malicious_transitions = malicious_episode_count * 2

    assert buffer.count_by_tag("malicious") == expected_malicious_transitions

    _, _, _, _, _, episode_ids, tags = buffer.sample_by_tag("malicious", batch_size=512)
    assert np.all(tags == "malicious")
    assert np.all((episode_ids >= 2000) & (episode_ids <= 3000))

    _, _, _, _, _, _, non_mal_tags = buffer.sample_excluding_tag("malicious", batch_size=512)
    assert np.all(non_mal_tags != "malicious")

    print("PASS: tagged replay buffer sampling works as expected.")


if __name__ == "__main__":
    test_tagged_sampling()
