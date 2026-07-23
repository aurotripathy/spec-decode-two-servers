docker pull furiosaai/furiosa-llm:latest

docker run -it --rm \
       --name draft-model \
       --device /dev/rngd:/dev/rngd \
       --security-opt seccomp=unconfined \
       --env HF_TOKEN=$HF_TOKEN \
       -v $HOME/.cache/huggingface:/root/.cache/huggingface \
       -p 8000:8000 \
       furiosaai/furiosa-llm:latest \
       serve furiosa-ai/Qwen2.5-0.5B-Instruct \
       --devices npu:0,npu:1,npu:2,npu:3 \
       --port 8000
