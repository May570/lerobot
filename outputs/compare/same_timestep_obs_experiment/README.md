# Same-Timestep Obs Experiment

这个实验用于验证：

- 从训练数据里选定一个 `episode` 和同一时刻 `t`
- 在线环境用对应初始场景 reset
- 若 `t > 0`，用离线轨迹动作推演到同一时刻
- 比较离线 `obs` 与在线 `obs`
- 重点查看：
  - `before_policy_pre`
  - `after_policy_pre`
  - `final_model_batch`

## 脚本

[run_same_timestep_obs_experiment.py](./run_same_timestep_obs_experiment.py)

## 默认输出目录

结果会保存在：

`/home/admin123/桌面/wjl/lerobot/outputs/compare/same_timestep_obs_experiment/runs/<run_name>/`

## 默认用法

```bash
conda run --no-capture-output -n lerobot_py312 \
python /home/admin123/桌面/wjl/lerobot/outputs/compare/same_timestep_obs_experiment/run_same_timestep_obs_experiment.py \
  --episode-index 0 \
  --frame-index 20
```

## 主要产物

- `metadata.json`
- `diff_summary.json`
- `offline_before_policy_pre.pt`
- `online_before_policy_pre.pt`
- `offline_after_policy_pre.pt`
- `online_after_policy_pre.pt`
- `offline_final_model_batch.pt`
- `online_final_model_batch.pt`
- `experiment_manifest.json`
- `viz_before_policy_pre/`（如果没有 `--skip-visualize`）
