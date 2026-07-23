# Client-Side Speculative Decoding Across Two LLM Servers

`spec-decode-two-servers.py` implements speculative decoding as a pure client, coordinating two independent OpenAI-compatible LLM servers over HTTP:

- **Server A (draft)** — a small model that proposes `K` tokens per round (`K` decode steps).
- **Server B (target)** — a large model that verifies all `K` drafted tokens in a **single prefill forward pass** using `prompt_logprobs`, and supplies the bonus/correction token from its own argmax.

Verification is greedy (`temperature=0`), so the committed output is **token-identical** to running the target model alone with greedy decoding — the same losslessness argument as classic speculative decoding in the greedy case.

## How it works

Each round:

1. The client asks the draft server for up to `K` greedy continuation tokens (a `/v1/completions` call with the context as token IDs).
2. The client sends `context + draft tokens` to the target server with `prompt_logprobs=1` and `max_tokens=1`. This is a single prefill pass that yields the target's argmax at every drafted position, plus one generated token.
3. Drafted tokens are accepted left-to-right while they match the target's argmax:
   - On the **first mismatch**, the target's rank-1 token at that position is committed as the correction, and the rest of the draft is dropped.
   - If **all `K` are accepted**, the target's single generated token is committed as the bonus token.
4. Accepted tokens are appended to the shared context and the next round begins. With prefix caching enabled on both servers, each round's prefill is incremental rather than from scratch.

All prompts are exchanged as **token IDs** (not text), so there is no tokenizer round-trip drift between the two servers.

## Requirements

1. Draft and target **must share the same tokenizer / vocabulary** (e.g. `Qwen2.5-0.5B-Instruct` drafting for `Qwen3-32B-FP8`).
2. Both servers must expose an OpenAI-compatible `/v1/completions` endpoint that supports `return_tokens_as_token_ids` and `prompt_logprobs` (recent vLLM or `furiosa-llm`).
3. Prefix caching enabled on both servers so the growing prompt is not re-prefilled from scratch every round.

Client-side dependencies:

```bash
pip install httpx transformers
```

## Launching the servers

Both servers run in the `furiosaai/furiosa-llm` Docker container, each pinned to half of the NPUs on the machine. See `launch-target.sh` and `launch-draft.sh`.

```bash
# Target (big) server: Qwen3-32B-FP8 on NPUs 4-7, port 8001
./launch-target.sh

# Draft (small) server: Qwen2.5-0.5B-Instruct on NPUs 0-3, port 8000
./launch-draft.sh
```

The model is sharded across the NPUs listed in `--devices` (e.g. `--devices npu:4,npu:5,npu:6,npu:7` for the target); there is no separate tensor-parallel flag. If your servers run vLLM on GPUs instead, the equivalent is `vllm serve <model> --tensor-parallel-size N`.

## Running the client

See `launch-driver.sh`:

```bash
python ./spec-decode-two-servers.py \
    --target-url http://0.0.0.0:8001 \
    --target-model furiosa-ai/Qwen3-32B-FP8 \
    --draft-url http://0.0.0.0:8000 \
    --draft-model furiosa-ai/Qwen2.5-0.5B-Instruct \
    --prompt "what is 100 plus 110?"
```

The generated text is printed to stdout once generation is complete (outside the timed region, so printing never affects the wall-time measurement); a stats summary is printed to stderr.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--target-url` / `--target-model` | (required) | Base URL and model name of the target (verifier) server |
| `--draft-url` / `--draft-model` | (required) | Base URL and model name of the draft (proposer) server |
| `--prompt` | (required) | The input prompt |
| `--chat` | off | Wrap the prompt with the model's chat template |
| `--k` | `5` | Draft tokens proposed per round |
| `--max-new-tokens` | `256` | Maximum tokens to generate |
| `--tokenizer` | `--target-model` | HF tokenizer name used client-side for encoding/decoding |
| `--stop-on-special` | off | Stop on any special token, not just `eos_token_id` |
| `--timeout` | `120.0` | HTTP timeout in seconds |
| `--verbose` | off | Log each round to stderr: drafted tokens, acceptance count, and the correction/bonus token |

## Output stats

At the end of a run, the client reports (on stderr):

- **tokens committed** — total tokens generated
- **rounds (target passes)** — number of verify calls, i.e. target forward passes
- **draft acceptance rate** — fraction of drafted tokens the target accepted
- **tokens per target pass** — average committed tokens per verify round (higher is better; `1.0` would mean speculation is buying nothing)
- **wall time** — generation time and throughput, with a breakdown of time spent in draft vs. verify calls. Only reported when `--verbose` is off, since per-round logging perturbs the timings.

## Tuning notes

- **`--k`** trades draft cost against verify efficiency. Larger `K` amortizes more target passes when acceptance is high, but wastes draft work when acceptance is low. Watch the acceptance rate and tokens-per-pass stats to tune it.
- Network latency matters: each round is one draft call plus one verify call, executed sequentially. Keep the client close to the servers.
