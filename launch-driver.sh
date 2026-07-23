# (2026.4.0b7) furiosa@furiosa:~/amd$ python spec-decode-two-servers.py
# usage: spec-decode-two-servers.py [-h] --target-url TARGET_URL
#                                   --target-model TARGET_MODEL
#                                   --draft-url DRAFT_URL
#                                   --draft-model DRAFT_MODEL
#                                   [--tokenizer TOKENIZER] --prompt PROMPT
#                                   [--chat] [--k K]
#                                   [--max-new-tokens MAX_NEW_TOKENS]
#                                   [--timeout TIMEOUT] [--stop-on-special]
#                                   [--quiet]

python ./spec-decode-two-servers.py \
       --target-url http://0.0.0.0:8001 \
       --target-model furiosa-ai/Qwen3-32B-FP8 \
       --draft-url http://0.0.0.0:8000 \
      --draft-model furiosa-ai/Qwen2.5-0.5B-Instruct \
      --prompt "what is 100 plus 110?"

