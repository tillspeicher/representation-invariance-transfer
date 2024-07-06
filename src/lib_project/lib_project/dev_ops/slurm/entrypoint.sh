#!/bin/bash

echo "Running job $SLURM_JOB_ID on $HOSTNAME"
echo "Starting at $(date)"
nvidia-smi
python3 --version

WRAPPER="$1"
USE_CPU_OFFLOAD="$2"
REMAINING_ARGS=("${@:3}")

NUM_GPUS=$(nvidia-smi -L | wc -l)
# MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_ADDR=localhost
# Use the job id to assign unique port numbers, to avoid conflicts
MASTER_PORT=$((29500 + ($SLURM_JOB_ID % 1000)))
is_port_in_use() {
  # Check if a process is listening on the specified port
  if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null; then
    return 0  # Port is in use
  else
    return 1  # Port is available
  fi
}
# Loop until a free port is found
while is_port_in_use $MASTER_PORT; do
  echo "Port $MASTER_PORT is in use, trying next port..."
  ((MASTER_PORT++))
done

if [ "$WRAPPER" == "docker" ]; then
    wrapper_command="./actions.py run-docker --no-progressbar"
elif [ "$WRAPPER" == "docker-devel" ]; then
    wrapper_command="./actions.py run-docker --no-progressbar --devel"
else
    wrapper_command="poetry run"
fi

env_command="env DISABLE_PROGRESSBAR=1"
if [ "$USE_CPU_OFFLOAD" == "1" ]; then
    env_command="${env_command} CPU_OFFLOAD=1"
fi

command="${wrapper_command} ${env_command} torchrun --nproc_per_node=$NUM_GPUS --nnodes 1 --master-addr $MASTER_ADDR --master-port $MASTER_PORT ${REMAINING_ARGS[@]}"
# command="$command torchrun --nproc_per_node=$NUM_GPUS --nnodes 1 ${REMAINING_ARGS[@]}"

echo "Running: $command"
# exec env DISABLE_PROGRESSBAR=1 ${command}
exec ${command}
echo "Done at $(date)"
