# Cluster & Tmux Commands Guide

This guide contains the essential commands for running long sweeps on the remote GPU cluster via SSH without jobs getting interrupted.

## The Core Concept: Tmux
When you connect via SSH and run a command, that command is tied to your connection. If your internet drops or you close your laptop, the command is killed.
`tmux` creates a "virtual terminal" on the remote machine that keeps running independently of your SSH connection.

## 1. Starting a Sweep
Always do this when running multi-hour jobs:

1. **Connect to the cluster:**
   ```bash
   ssh gpu1
   ```
2. **Start a new tmux session:** (name it whatever you like, e.g., `scaling_run`)
   ```bash
   tmux new -s scaling_run
   ```
3. **Navigate and launch the script:**
   ```bash
   cd ~/UncertaintyV1/nn_decoder
   
   # For Population Scaling Sweeps:
   PY=~/cluster-env/.venv/bin/python bash run_all_scaling.sh
   
   # Or for Hyperparameter Sweeps:
   PY=~/cluster-env/.venv/bin/python bash run_all_sweeps.sh
   ```
4. **Safely Detach:** 
   Press `Ctrl+B`, release both, then press `D`. You are now safely detached and can close your terminal or laptop.

## 2. Checking Progress

You can check on your jobs at any time from your Mac.

**Option A: Reattach to the tmux session (Interactive)**
```bash
ssh gpu1
tmux attach -t scaling_run
```
*(To leave it running and exit, use `Ctrl+B` then `D` again!)*

**Option B: Tail the logs without attaching**
```bash
# See the latest output of a specific mouse
ssh gpu1 "tail -f ~/UncertaintyV1/nn_decoder/neuron_scaling_logs/Q_mouse0.log"
```
*(Press `Ctrl+C` to stop watching the log)*

## 3. Other Useful Tmux Commands

* **List all running sessions:** `tmux ls`
* **Kill a specific session (stops the jobs inside):** `tmux kill-session -t scaling_run`
* **Scroll up in tmux:** Press `Ctrl+B`, then `[` to enter "scroll mode". Use arrow keys or Page Up/Down. Press `q` to exit scroll mode.

## FAQ

**Q: What happens if I close my Mac or terminal while still "attached"?**
Nothing bad! If your SSH connection drops unexpectedly (closed terminal, lost Wi-Fi, Mac went to sleep), `tmux` detects the broken connection and automatically and safely detaches your session on the remote server. Your jobs continue running exactly as if you had detached manually.

**Q: How do I know if something goes wrong once I've detached?**
1. **Check the tmux window later:** The bash scripts (`run_all_scaling.sh`, etc.) collect the exit codes of all parallel processes. If something fails, the script will print `FAIL` and tell you exactly which log file to look at.
2. **Check the logs for errors:** You can run a command like this to scan all logs for failures:
   ```bash
   ssh gpu1 "grep -i 'fail\|error' ~/UncertaintyV1/nn_decoder/neuron_scaling_logs/*.log"
   ```
