from huggingface_hub import hf_hub_download, snapshot_download

# # 模型名称（替换成你需要下载的模型名称）
# model_name = "lerobot/pi0"

# # 下载整个仓库
# repo_path = snapshot_download(repo_id=model_name)

# print(f"Model repository downloaded to: {repo_path}")


# 数据集名称
dataset_name = "lerobot/aloha_sim_transfer_cube_human"

# 下载整个数据集仓库（默认缓存到 ~/.cache/huggingface/hub）
local_dir = snapshot_download(repo_id=dataset_name, repo_type="dataset")

print(f"Dataset downloaded to: {local_dir}")

