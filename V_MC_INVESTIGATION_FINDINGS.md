# Investigation: Where `v_mc` Was Introduced

## Quick Answer

**Q: Where was `v_mc` introduced?**

**A:** The `v_mc` variable was introduced in commit **`6583257a47f6195c2fc926797d55c6f47f90733a`** on **Feb 8, 2019** in file **`_under_construction/week4_approx_rl/utils.py`** (later moved to `week04_approx_rl/utils.py`).

**Q: Why is it missing now?**

**A:** In commit **`71a5612a4151aa645a7a28732f84b755ae886995`** on **Mar 13, 2024**, the entire `utils.py` file was deleted during a major restructure. The function was replaced with a simplified version in `week04_approx_rl/dqn/analysis.py` that doesn't compute `v_mc` or `v_agent`.

**Q: How do I view the original code?**

**A:** See the Appendix at the bottom of this file, or run: `git show 71a5612~1:week04_approx_rl/utils.py`

---

## Summary

The `v_mc` variable was introduced in the original `utils.py` file that is **no longer present in the current repository**. The current code in `week04_approx_rl/dqn/analysis.py` does not compute or return `v_mc` and `v_agent`, which is why the notebooks reference these fields but they are not available.

## Commit Details

### Original Introduction
- **Commit SHA**: `6583257a47f6195c2fc926797d55c6f47f90733a`
- **Author**: zshrav <40326977+zshrav@users.noreply.github.com>
- **Date**: Fri Feb 8 20:27:44 2019 +0500
- **Message**: "Add files via upload"
- **File**: `_under_construction/week4_approx_rl/utils.py` (later moved to `week04_approx_rl/utils.py`)

### Publication to Main Directory
- **Commit SHA**: `63245f67a3d50a9a7120a9a8eb310e73e566d64b`
- **Author**: zshrav <zshrav@yandex.ru>
- **Date**: Mon Feb 18 21:33:10 2019 +0500
- **Message**: "publish week4"
- **Action**: Moved files from `_under_construction/week4_approx_rl/` to `week04_approx_rl/`

## The Original Implementation

The original `play_and_log_episode` function in `utils.py` (commit `6583257`) included:

```python
def play_and_log_episode(env, agent, gamma=0.99, t_max=10000):
    """
    always greedy
    """
    states = []
    v_mc = []
    v_agent = []
    q_spreads = []
    td_errors = []
    rewards = []

    s = env.reset()
    for step in range(t_max):
        states.append(s)
        qvalues = agent.get_qvalues([s])
        max_q_value, min_q_value = np.max(qvalues), np.min(qvalues)
        v_agent.append(max_q_value)
        q_spreads.append(max_q_value - min_q_value)
        if step > 0:
            td_errors.append(np.abs(rewards[-1] + gamma * v_agent[-1] - v_agent[-2]))

        action = qvalues.argmax(axis=-1)[0]

        s, r, done, _ = env.step(action)
        rewards.append(r)
        if done:
            break
    td_errors.append(np.abs(rewards[-1] + gamma * v_agent[-1] - v_agent[-2]))

    v_mc = get_cum_discounted_rewards(rewards, gamma)

    return_pack = {
        'states': np.array(states),
        'v_mc': np.array(v_mc),
        'v_agent': np.array(v_agent),
        'q_spreads': np.array(q_spreads),
        'td_errors': np.array(td_errors),
        'rewards': np.array(rewards),
        'episode_finished': np.array(done)
    }

    return return_pack
```

## Key Findings

1. **`v_mc` (Monte Carlo Value)**: Computed using `get_cum_discounted_rewards(rewards, gamma)`, which calculates cumulative discounted rewards: `r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...`

2. **`v_agent` (Agent Value)**: The maximum Q-value predicted by the agent for each state: `max_q_value = np.max(qvalues)`

3. **The function also computed**:
   - `q_spreads`: The difference between max and min Q-values
   - `td_errors`: Temporal difference errors

## Current State

The current implementation in `week04_approx_rl/dqn/analysis.py` (added in commit `5e1d487c194e7f154a45f98ffa495d39935cf9bc` on Sun Sep 7 14:47:43 2025) is a simplified version that does **NOT** include:
- `v_mc` calculation
- `v_agent` calculation  
- `q_spreads` calculation
- `td_errors` calculation
- The `gamma` parameter

The notebooks `homework_tf.ipynb` and `homework_pytorch_debug.ipynb` still reference these missing fields, which causes the issue described.

## What Happened

The repository was restructured in a major update:

- **Commit SHA**: `71a5612a4151aa645a7a28732f84b755ae886995`
- **Author**: zshrav <zshrav@yandex.ru>
- **Date**: Wed Mar 13 04:18:31 2024 +0500
- **Message**: "spring_2024_week_04_approx_rl: a major update"

In this commit, the original `week04_approx_rl/utils.py` file was **deleted** and replaced with:
- A new `week04_approx_rl/dqn/` module structure
- A simplified `week04_approx_rl/dqn/analysis.py` with a reduced version of `play_and_log_episode`
- A much simpler `week04_approx_rl/dqn/utils.py` that only contains `is_enough_ram()` and `linear_decay()` functions

The notebooks (`homework_tf.ipynb` and `homework_pytorch_debug.ipynb`) were not updated to reflect this change, so they still reference fields that are no longer computed.

## Complete Timeline

1. **Feb 8, 2019** (commit `6583257`): Original `utils.py` created with full `play_and_log_episode` function including `v_mc` and `v_agent` calculations
2. **Feb 18, 2019** (commit `63245f6`): Files moved from `_under_construction/` to `week04_approx_rl/`
3. **Mar 13, 2024** (commit `71a5612`): Major restructure - `utils.py` deleted, new `dqn/` module created with simplified functions
4. **Sep 7, 2025** (commit `5e1d487`): Repository shallow cloned/merged (current state)

## The Missing Function: `get_cum_discounted_rewards`

This helper function was also in the original `utils.py` and is crucial for computing `v_mc`:

```python
def get_cum_discounted_rewards(rewards, gamma):
    """
    evaluates cumulative discounted rewards:
    r_t + gamma * r_{t+1} + gamma^2 * r_{t_2} + ...
    """
    cum_rewards = []
    cum_rewards.append(rewards[-1])
    for r in reversed(rewards[:-1]):
        cum_rewards.insert(0, r + gamma * cum_rewards[0])
    return cum_rewards
```

## How to Fix

To make the notebooks work again, the `play_and_log_episode` function in `week04_approx_rl/dqn/analysis.py` needs to be updated to restore the missing functionality:

1. Add back the `gamma` parameter
2. Compute `v_agent` as the max Q-value for each state
3. Compute `v_mc` using the `get_cum_discounted_rewards` helper function
4. Optionally restore `q_spreads` and `td_errors` if needed by the notebooks

The last working version of the function before deletion can be found at commit `71a5612~1` (parent of the major update commit).

## Appendix: Complete Last Version of utils.py

Here is the complete content of `week04_approx_rl/utils.py` before it was deleted (from commit `71a5612~1`):

```python
import numpy as np
import psutil
from scipy.signal import fftconvolve, gaussian


def get_cum_discounted_rewards(rewards, gamma):
    """
    evaluates cumulative discounted rewards:
    r_t + gamma * r_{t+1} + gamma^2 * r_{t_2} + ...
    """
    cum_rewards = []
    cum_rewards.append(rewards[-1])
    for r in reversed(rewards[:-1]):
        cum_rewards.insert(0, r + gamma * cum_rewards[0])
    return cum_rewards


def play_and_log_episode(env, agent, gamma=0.99, t_max=10000):
    """
    always greedy
    """
    states = []
    v_mc = []
    v_agent = []
    q_spreads = []
    td_errors = []
    rewards = []

    s, _ = env.reset()
    for step in range(t_max):
        states.append(s)
        qvalues = agent.get_qvalues([s])
        max_q_value, min_q_value = np.max(qvalues), np.min(qvalues)
        v_agent.append(max_q_value)
        q_spreads.append(max_q_value - min_q_value)
        if step > 0:
            td_errors.append(
                np.abs(rewards[-1] + gamma * v_agent[-1] - v_agent[-2]))

        action = qvalues.argmax(axis=-1)[0]

        s, r, terminated, truncated, _ = env.step(action)
        rewards.append(r)
        if terminated or truncated:
            break
    td_errors.append(np.abs(rewards[-1] + gamma * v_agent[-1] - v_agent[-2]))

    v_mc = get_cum_discounted_rewards(rewards, gamma)

    return_pack = {
        'states': np.array(states),
        'v_mc': np.array(v_mc),
        'v_agent': np.array(v_agent),
        'q_spreads': np.array(q_spreads),
        'td_errors': np.array(td_errors),
        'rewards': np.array(rewards),
        'episode_finished': np.array(terminated or truncated)
    }

    return return_pack


def img_by_obs(obs, state_dim):
    """
    Unwraps obs by channels.
    observation is of shape [c, h=w, w=h]
    """
    return obs.reshape([-1, state_dim[2]])


def is_enough_ram(min_available_gb=0.1):
    mem = psutil.virtual_memory()
    return mem.available >= min_available_gb * (1024 ** 3)


def linear_decay(init_val, final_val, cur_step, total_steps):
    if cur_step >= total_steps:
        return final_val
    return (init_val * (total_steps - cur_step) +
            final_val * cur_step) / total_steps


def smoothen(values):
    kernel = gaussian(100, std=100)
    # kernel = np.concatenate([np.arange(100), np.arange(99, -1, -1)])
    kernel = kernel / np.sum(kernel)
    return fftconvolve(values, kernel, 'valid')
```
