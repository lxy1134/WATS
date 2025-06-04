#!/bin/bash

calibration_methods=("WATS")

yaml_directory="./config"
log_directory="./log"

mkdir -p "$log_directory"

yaml_files=("reddit.yaml") # "reddit.yaml" "ogbn-arxiv.yaml""cs.yaml" "physics.yaml" "citeseer.yaml" "computers.yaml" "cora-full.yaml" "cora.yaml" "photo.yaml"  "pubmed.yaml"


# 检查 yq 是否可用
if ! command -v yq &> /dev/null; then
    echo "Error: yq not found in PATH. Please make sure it's installed and in your PATH."
    exit 1
fi

run_script() {
  
  local yaml_file="$1"
  local method="$2"
  local gpu_id=0  # 固定为 GPU 0

  dataset_name=$(basename "$yaml_file" .yaml)

  yq eval ".calibration.calibrator_name = \"$method\"" -i "$yaml_directory/$yaml_file"

  log_file="$log_directory/${dataset_name}_${method}.txt"

  echo "Running: python main.py --dataset=$dataset_name --gpu=$gpu_id --n_runs=10 with $method on GPU $gpu_id"
  CUDA_VISIBLE_DEVICES=$gpu_id python main.py --dataset="$dataset_name" --gpu="$gpu_id" --n_runs=10 > "$log_file" 2>&1
}

# 顺序跑所有组合
for yaml_file in "${yaml_files[@]}"
do
  for method in "${calibration_methods[@]}"
  do
    run_script "$yaml_file" "$method"
  done
done