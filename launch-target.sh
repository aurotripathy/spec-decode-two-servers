docker pull furiosaai/furiosa-llm:latest

docker run -it --rm \
       --name target-model \
       --device /dev/rngd:/dev/rngd \
       --security-opt seccomp=unconfined \
       --env HF_TOKEN=$HF_TOKEN \
       -v $HOME/.cache/huggingface:/root/.cache/huggingface \
       -p 8001:8001 \
       furiosaai/furiosa-llm:latest \
       serve furiosa-ai/Qwen3-32B-FP8 \
       --devices npu:4,npu:5,npu:6,npu:7 \
       --port 8001
