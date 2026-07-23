
python ./spec-decode-two-servers.py \
       --target-url http://0.0.0.0:8001 \
       --target-model furiosa-ai/Qwen3-32B-FP8 \
       --draft-url http://0.0.0.0:8000 \
       --draft-model furiosa-ai/Qwen2.5-0.5B-Instruct \
       --prompt "what is 100 plus 110?" \
       --chat \
       --k 5 \
       --verbose \
       --max-new-tokens 1000


