# Investigation Summary: v_mc Variable

## The Question

> "where v_mc was introduced initially? as I don't see it in play_and_log_episode and can't find it being calculated anywhere. find commit and file where it were introduced"

## The Answer

### Quick Facts
- **First introduced**: Commit `6583257a47f6195c2fc926797d55c6f47f90733a` on February 8, 2019
- **Original file**: `_under_construction/week4_approx_rl/utils.py`
- **Removed in**: Commit `71a5612a4151aa645a7a28732f84b755ae886995` on March 13, 2024
- **Why it's missing**: Major code restructure deleted the original `utils.py` file

### What is v_mc?

`v_mc` stands for "Monte Carlo Value" - it's the cumulative discounted reward calculated from actual episode rewards:

```python
v_mc = get_cum_discounted_rewards(rewards, gamma)
```

This computes: `r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...`

### Where to Learn More

1. **[V_MC_INVESTIGATION_FINDINGS.md](./V_MC_INVESTIGATION_FINDINGS.md)** - Complete investigation details with code examples
2. **[INVESTIGATION_LINKS.md](./INVESTIGATION_LINKS.md)** - Direct links to GitHub commits and files

### View the Original Code

To see the complete implementation with `v_mc`:

```bash
git show 70124e39d6a0c3e38a0bbeb71bcaa4c5b1e8fa90:week04_approx_rl/utils.py
```

Or view it on GitHub: [utils.py before deletion](https://github.com/g0djan/Practical_RL/blob/70124e39d6a0c3e38a0bbeb71bcaa4c5b1e8fa90/week04_approx_rl/utils.py)

### The Problem

The Jupyter notebooks (`homework_tf.ipynb` and `homework_pytorch_debug.ipynb`) still try to access `record['v_mc']` and `record['v_agent']`, but the current `play_and_log_episode` function in `week04_approx_rl/dqn/analysis.py` doesn't compute these values anymore.

### Timeline

1. **Feb 8, 2019**: Original `utils.py` created with full functionality
2. **Feb 18, 2019**: Files published from `_under_construction/` to `week04_approx_rl/`
3. **Mar 13, 2024**: Major restructure - `utils.py` deleted, simplified version created
4. **Current**: Notebooks reference missing variables

---

**Note**: This investigation was conducted by examining the full git history after unshallowing the repository.
