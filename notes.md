Ashish slack thread https://redhat-internal.slack.com/archives/C0BKVKSANN4/p1785270169894349

In that thread:
docker run --gpus all \
  --privileged --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e GLOO_SOCKET_IFNAME=$IFACE_NAME \
  -e NCCL_SOCKET_IFNAME=$IFACE_NAME \
  -e VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1 \
  vllm/vllm-openai:kimi-k3 moonshotai/Kimi-K3 \
  --trust-remote-code \
  --load-format fastsafetensors \
  --gpu-memory-utilization 0.95 \
  -cc.pass_config.fuse_allreduce_rms=False \
  --tensor-parallel-size 16 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr $HEAD_IP \
  --no-enable-flashinfer-autotune \
  --moe-backend marlin \
  --disable-custom-all-reduce \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3