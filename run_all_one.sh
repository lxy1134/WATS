#!/usr/bin/env bash
set -u pipefail # 去掉了 -e，防止 Python 报错导致脚本中断

# --- 配置区 ---
calibration_methods=("GETS")
yaml_directory="./config"
log_directory="./log"
yaml_files=("roman.yaml" "tolokers.yaml" "reddit.yaml" "citeseer.yaml" "computers.yaml" "cora-full.yaml" "cora.yaml" "photo.yaml" "pubmed.yaml")

mkdir -p "$log_directory"

# --- yq 版本检测 ---
need_yq=true
YQ_MODE="unknown"

if command -v yq >/dev/null 2>&1; then
  YQ_VER=$(yq --version 2>&1 || true)
  if echo "$YQ_VER" | grep -qi 'mikefarah/yq'; then
    if echo "$YQ_VER" | grep -qE 'version 4\.'; then
      YQ_MODE="v4"
    else
      YQ_MODE="v3"
    fi
  else
    need_yq=false
    echo "WARNING: yq found but not mikefarah/yq. Falling back to sed."
  fi
else
  need_yq=false
  echo "WARNING: yq not found. Falling back to sed."
fi

write_calibrator() {
  local cfg="$1" ; local m="$2"

  if [ ! -f "$cfg" ]; then
    echo "ERROR: config not found: $cfg" >&2
    exit 1
  fi

  # 1. 修改配置
  if [ "${need_yq}" = true ]; then
    if [ "$YQ_MODE" = "v4" ]; then
      yq e -i ".calibration.calibrator_name = \"$m\"" "$cfg"
    else
      yq w -i "$cfg" "calibration.calibrator_name" "$m"
    fi
  else
    # 兜底 sed：尽力只替换 key 为 calibrator_name 的行
    sed -i -E 's/^([[:space:]]*calibrator_name:[[:space:]]*).*/\1'"$m"'/' "$cfg"
  fi

  # 2. 验证修改 (Check)
  local read_back="__NA__"
  if [ "${need_yq}" = true ]; then
    if [ "$YQ_MODE" = "v4" ]; then
      # 注意：使用 -r 也就是 raw output，防止输出带引号 (如 "WATS") 导致对比失败
      read_back=$(yq e -r '.calibration.calibrator_name // "__NULL__"' "$cfg")
    else
      read_back=$(yq r "$cfg" "calibration.calibrator_name")
      [ -z "$read_back" ] && read_back="__NULL__"
    fi
  else
    # grep 粗读
    read_back=$(grep -E '^[[:space:]]*calibrator_name:' "$cfg" | tail -n1 | awk -F: '{sub(/^[ ]+/,"",$2); print $2}')
    read_back=${read_back:-__NULL__} # remove potential carriage returns or spaces handled by awk but just in case
    # trim spaces
    read_back=$(echo "$read_back" | xargs)
  fi

  if [ "$read_back" != "$m" ]; then
    echo "ERROR: calibrator_name update failed! File: $(basename "$cfg"). Got '$read_back', expected '$m'." >&2
    # 这里依然可以选择 exit，因为配置改不成功，跑也没意义
    exit 1
  fi
}

run_script() {
  local yaml_file="$1"
  local method="$2"
  local gpu_id=0 # 如果需要动态指定 GPU，可以在这里修改

  local cfg_path="$yaml_directory/$yaml_file"
  local dataset_name="${yaml_file%.yaml}"
  local log_file="$log_directory/${dataset_name}_${method}.txt"

  # 先修改配置
  write_calibrator "$cfg_path" "$method"

  echo "[$(date '+%H:%M:%S')] Running: Dataset=$dataset_name | Method=$method"
  
  # 执行 Python
  # 注意：这里使用了 if 来捕获错误，而不是让脚本直接崩掉
  if CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
    --dataset "$dataset_name" --gpu "$gpu_id" --n_runs 10 \
    >"$log_file" 2>&1; then
      echo "    -> Success. Log: $log_file"
  else
      echo "    -> FAILED! Check log: $log_file"
      # 可选：如果不希望失败中断后续循环，这里什么都不做
      # 如果希望失败一个数据集就跳过这个数据集的其他方法，可以在这里做逻辑处理
  fi
}

# --- 主循环 ---
echo "Starting experiments..."
echo "Config Mode: Using ${need_yq:+yq ($YQ_MODE)}${need_yq:-sed fallback}"

for yaml_file in "${yaml_files[@]}"; do
  for method in "${calibration_methods[@]}"; do
    run_script "$yaml_file" "$method"
  done
done

echo "All experiments finished."