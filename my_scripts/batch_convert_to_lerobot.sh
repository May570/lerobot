# #!/usr/bin/env bash
# set -e

# # ====== 固定路径 ======
# HF_LEROBOT_HOME=/share/project/wujiling/datasets
# PROCESSED_ROOT=/share/project/wujiling/datasets/processed_data
# SCRIPT=python
# CONVERTER=scripts/convert_aloha_data_to_lerobot_robotwin.py

# export HF_LEROBOT_HOME

# echo "HF_LEROBOT_HOME = $HF_LEROBOT_HOME"
# echo "Processed data root = $PROCESSED_ROOT"
# echo

# # ====== 遍历所有 *-demo_clean-50 目录 ======
# for dir in "$PROCESSED_ROOT"/*-demo_clean-50; do
#     [ -d "$dir" ] || continue

#     base=$(basename "$dir")
#     echo "Processing: $base"

#     # base = adjust_bottle-demo_clean-50
#     # task = adjust_bottle
#     task=${base%%-demo_clean-50}

#     # repo_id = clean/adjust_bottle_demo
#     repo_id="clean/${task}_demo"

#     echo "  raw-dir : $dir"
#     echo "  repo-id : $repo_id"

#     $SCRIPT "$CONVERTER" \
#         --raw-dir "$dir" \
#         --repo-id "$repo_id" \
#         --mode image

#     echo "✔ Done: $repo_id"
#     echo
# done

# echo "All tasks converted."

#!/usr/bin/env bash
set -e

# ====== 固定路径 ======
HF_LEROBOT_HOME=/share/project/wujiling/datasets
PROCESSED_ROOT=/share/project/wujiling/datasets/processed_data
CONVERTER=scripts/convert_aloha_data_to_lerobot_robotwin.py

export HF_LEROBOT_HOME

# ====== 多任务合并到一个 repo_id ======
REPO_ID="multi/multitask_demo"

echo "HF_LEROBOT_HOME = $HF_LEROBOT_HOME"
echo "Processed data root = $PROCESSED_ROOT"
echo "Repo id = $REPO_ID"
echo

python "$CONVERTER" \
  --raw-dir "$PROCESSED_ROOT" \
  --repo-id "$REPO_ID" \
  --mode image

echo "✔ Done: $REPO_ID"