#!/bin/bash
set -e

# 获取脚本所在目录，动态设置路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH="${SCRIPT_DIR}"

cd "${REPO_PATH}"

# 激活虚拟环境（使用指定路径）
VENV_PATH="${VENV_PATH:-/inspire/hdd/global_user/zhangchenxi-253108310322/RLinf/.venv}"
source "${VENV_PATH}/bin/activate"

# 1. 关键：不要连平台自带 Ray
unset RAY_ADDRESS

# 2. 映射平台的 rank
NODE_RANK=${PET_NODE_RANK:-${RANK:-0}}
export RLINF_NODE_RANK=${NODE_RANK}
export RANK=${NODE_RANK}

# 3. 使用平台提供的 MASTER_ADDR，但 Ray 用独立端口（避免和 DDP 冲突）
RAY_HEAD_ADDR=${MASTER_ADDR:-${PET_MASTER_ADDR}}
RAY_PORT=${RAY_PORT:-6379}  # Ray 默认端口，不用 MASTER_PORT

echo "[RLinf] NODE_RANK=${NODE_RANK}, RLINF_NODE_RANK=${RLINF_NODE_RANK}"
echo "[RLinf] RAY_HEAD_ADDR=${RAY_HEAD_ADDR}, RAY_PORT=${RAY_PORT}"

# 4. 停止可能存在的旧 Ray 实例
ray stop 2>/dev/null || true

# 5. 启动/加入 Ray 集群
if [ "${NODE_RANK}" -eq 0 ]; then
    # 头节点：启动 Ray head
    echo "[RLinf] Starting Ray head node..."

    if [ -n "${RAY_HEAD_ADDR}" ]; then
        ray start --head \
            --port=${RAY_PORT} \
            --node-ip-address=${RAY_HEAD_ADDR} \
            --memory=461708984320 \
            --dashboard-host=0.0.0.0
    else
        ray start --head \
            --port=${RAY_PORT} \
            --memory=461708984320 \
            --dashboard-host=0.0.0.0
    fi

    echo "[RLinf] Ray head started"
else
    # 工作节点：连接到头节点
    if [ -z "${RAY_HEAD_ADDR}" ]; then
        echo "[RLinf] ERROR: MASTER_ADDR not set, cannot connect to head node"
        exit 1
    fi

    echo "[RLinf] Connecting to Ray head at ${RAY_HEAD_ADDR}:${RAY_PORT}..."

    # 先等待头节点启动（30秒）
    echo "[RLinf] Waiting 30s for head node to start..."
    sleep 30

    # 重试连接（最多120秒）
    CONNECTED=false
    for i in {1..120}; do
        if ray start --address="${RAY_HEAD_ADDR}:${RAY_PORT}" \
            --memory=461708984320 2>&1; then
            echo "[RLinf] Successfully connected to Ray cluster"
            CONNECTED=true
            break
        fi
        echo "[RLinf] Connection attempt ${i}/60 failed, retrying..."
        ray stop 2>/dev/null || true
        sleep 1
    done

    if [ "${CONNECTED}" = "false" ]; then
        echo "[RLinf] ERROR: Failed to connect to Ray head after 120 attempts"
        exit 1
    fi
fi

# 6. 等待所有节点就绪（只在头节点检查）
if [ "${NODE_RANK}" -eq 0 ]; then
    echo "[RLinf] Waiting for all nodes to join..."
    NNODES=${PET_NNODES:-4}

    for i in {1..120}; do
        # 使用 ray status 解析节点数
        NUM_NODES=$(ray status 2>/dev/null | grep -c "node_id" || echo "0")
        NUM_NODES=$(echo "${NUM_NODES}" | tr -d '[:space:]')

        # 确保 NUM_NODES 是纯数字
        if ! [[ "${NUM_NODES}" =~ ^[0-9]+$ ]]; then
            NUM_NODES=0
        fi

        echo "[RLinf] Check ${i}/120: ${NUM_NODES}/${NNODES} nodes joined"

        if [ "${NUM_NODES}" -ge "${NNODES}" ]; then
            echo "[RLinf] All ${NNODES} nodes are ready!"
            break
        fi

        if [ $i -eq 120 ]; then
            echo "[RLinf] WARNING: Only ${NUM_NODES}/${NNODES} nodes joined, continuing anyway..."
            echo "[RLinf] Current Ray cluster status:"
            ray status 2>/dev/null | head -20 || true
        fi

        sleep 2
    done
fi

# 7. 只在头节点启动训练
if [ "${NODE_RANK}" -eq 0 ]; then
    echo "[RLinf] Launching training script..."
    bash examples/embodiment/run_embodiment.sh egrpo
else
    echo "[RLinf] Worker node, keeping alive..."
    # Worker 节点保持活跃
    while true; do
        sleep 600
    done
fi
