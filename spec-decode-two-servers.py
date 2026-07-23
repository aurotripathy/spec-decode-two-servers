#!/usr/bin/env python3
"""
Client-side speculative decoding across TWO separate OpenAI-compatible
LLM servers.

  Server A (draft) : small model, proposes K tokens per round (K decode steps).
  Server B (target): large model, verifies all K drafted tokens in a SINGLE
                     prefill forward pass via `prompt_logprobs`, and supplies
                     the bonus/correction token from its own argmax.

Verification is greedy (temperature=0), so the committed output is
token-identical to running the target model alone with greedy decoding
(same losslessness argument as classic speculative decoding, greedy case).

Hard requirements
-----------------
1. Draft and target MUST share the same tokenizer / vocabulary
   (e.g. Qwen2.5-0.5B-Instruct drafting for Qwen3-32B-FP8).
2. Both servers must support `return_tokens_as_token_ids` and
   `prompt_logprobs` on /v1/completions (recent vLLM or furiosa-llm).
3. Prefix caching ON so the growing prompt is not re-prefilled from
   scratch every round on either server.

Example server launches
-----------------------
  # Target (big) server: Qwen3-32B-FP8 on NPUs 4-7, port 8001
  ./launch-target.sh

  # Draft (small) server: Qwen2.5-0.5B-Instruct on NPUs 0-3, port 8000
  ./launch-draft.sh

Run (see launch-driver.sh)
--------------------------
  pip install httpx transformers

  python ./spec-decode-two-servers.py \
      --target-url http://0.0.0.0:8001 \
      --target-model furiosa-ai/Qwen3-32B-FP8 \
      --draft-url http://0.0.0.0:8000 \
      --draft-model furiosa-ai/Qwen2.5-0.5B-Instruct \
      --prompt "what is 100 plus 110?"
"""

import argparse
import asyncio
import sys
import time

import httpx
from transformers import AutoTokenizer

from utils import fetch_model_id, parse_token_id


async def draft_propose(client: httpx.AsyncClient, url: str, model: str,
                        ids: list[int], k: int) -> list[int]:
    """Ask the draft server for up to k greedy continuation tokens."""
    r = await client.post(f"{url}/v1/completions", json={
        "model": model,
        "prompt": ids,                     # token IDs -> no tokenizer drift
        "max_tokens": k,
        "temperature": 0,
        "logprobs": 0,                     # we only need the sampled tokens
        "return_tokens_as_token_ids": True,
    })
    r.raise_for_status()
    data = r.json()
    lp = data["choices"][0].get("logprobs") or {}
    toks = lp.get("tokens") or []
    return [parse_token_id(t) for t in toks]


async def target_verify(client: httpx.AsyncClient, url: str, model: str,
                        ctx_ids: list[int], draft_ids: list[int]
                        ) -> tuple[int, int]:
    """
    One prefill pass on the target over (context + draft tokens).

    Returns (n_accepted, next_token):
      - n_accepted: how many leading draft tokens match the target argmax
      - next_token: if all K accepted -> the target's bonus token
                    (from the max_tokens=1 generation);
                    on first mismatch -> the target's correction token
                    (the rank-1 token at the rejected position).
    """
    r = await client.post(f"{url}/v1/completions", json={
        "model": model,
        "prompt": ctx_ids + draft_ids,
        "max_tokens": 1,                   # bonus token if everything accepted
        "temperature": 0,
        "logprobs": 0,
        "prompt_logprobs": 1,              # per-position top-1 over the prompt
        "return_tokens_as_token_ids": True,
    })
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]

    plp = choice.get("prompt_logprobs") or data.get("prompt_logprobs")
    if plp is None:
        raise RuntimeError("Server did not return prompt_logprobs. "
                           "The target server must support prompt_logprobs "
                           "(recent vLLM or furiosa-llm).")

    base = len(ctx_ids)
    n_accepted = 0
    for j, tok in enumerate(draft_ids):
        entry = plp[base + j]              # dist over position base+j given prefix
        if entry is None:
            raise RuntimeError(f"Missing prompt_logprobs entry at {base + j}")
        # entry: {token_id_str: {"logprob":..., "rank":..., "decoded_token":...}}
        argmax_id = None
        for tid_str, info in entry.items():
            if info.get("rank") == 1:
                argmax_id = int(tid_str)
                break
        if argmax_id is None:
            raise RuntimeError(f"No rank-1 token in prompt_logprobs at {base + j}")
        if argmax_id == tok:
            n_accepted += 1
        else:
            return n_accepted, argmax_id   # correction token, rest of draft dropped

    # All drafted tokens accepted -> bonus token from the single generated step.
    bonus = parse_token_id(choice["logprobs"]["tokens"][0])
    return n_accepted, bonus


async def generate(args) -> None:
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.target_model)

    if args.chat:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True, tokenize=True, return_dict=False)
    else:
        ids = tok(args.prompt).input_ids

    eos_ids = set(tok.all_special_ids if args.stop_on_special else [tok.eos_token_id])

    committed: list[int] = []
    rounds = drafted_total = accepted_total = 0
    t_draft = t_verify = 0.0

    print("Generation Started:", file=sys.stderr, flush=True)
    t0 = time.perf_counter()

    limits = httpx.Limits(max_keepalive_connections=4)
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
        done = False
        while not done and len(committed) < args.max_new_tokens:
            k = min(args.k, args.max_new_tokens - len(committed))

            ts = time.perf_counter()
            draft_ids = await draft_propose(
                client, args.draft_url, args.draft_model, ids, k)
            t_draft += time.perf_counter() - ts

            if not draft_ids:
                print("\n[warn] draft server returned no tokens; stopping.",
                      file=sys.stderr)
                break

            ts = time.perf_counter()
            n_acc, next_tok = await target_verify(
                client, args.target_url, args.target_model, ids, draft_ids)
            t_verify += time.perf_counter() - ts

            rounds += 1
            drafted_total += len(draft_ids)
            accepted_total += n_acc

            if args.verbose:
                outcome = "bonus" if n_acc == len(draft_ids) else "correction"
                print(f"\n[round {rounds}] drafted {len(draft_ids)}: "
                      f"{tok.decode(draft_ids)!r} | accepted {n_acc} | "
                      f"{outcome}: {tok.decode([next_tok])!r} "
                      f"(id {next_tok})", file=sys.stderr)

            new_tokens = draft_ids[:n_acc] + [next_tok]
            kept_this_round: list[int] = []
            for t in new_tokens:
                if t in eos_ids:
                    done = True
                    break
                committed.append(t)
                kept_this_round.append(t)
                if len(committed) >= args.max_new_tokens:
                    done = True
                    break

            # Grow the shared context; prefix caching on both servers makes
            # the next round's prefill incremental rather than from-scratch.
            ids = ids + kept_this_round

        wall = time.perf_counter() - t0
        print("Generation Stopped:", file=sys.stderr, flush=True)

        draft_id, target_id = await asyncio.gather(
            fetch_model_id(client, args.draft_url),
            fetch_model_id(client, args.target_url))

    # Decode and print outside the timed region so output cost never
    # pollutes the wall-time measurement.
    text = tok.decode(committed, skip_special_tokens=True)
    print(text)

    n = len(committed)
    print("\n" + "-" * 60, file=sys.stderr)
    print(f"tokens committed        : {n}", file=sys.stderr)
    print(f"rounds (target passes)  : {rounds}", file=sys.stderr)
    print(f"draft acceptance rate   : "
          f"{(accepted_total / drafted_total * 100 if drafted_total else 0):.1f}% "
          f"({accepted_total}/{drafted_total})", file=sys.stderr)
    print(f"tokens per target pass  : {(n / rounds if rounds else 0):.2f}",
          file=sys.stderr)
    if not args.verbose:
        # Per-round logging perturbs the timings, so only report them
        # when verbose is off.
        print(f"wall time               : {wall:.2f}s "
              f"({n / wall if wall > 0 else 0:.1f} tok/s)", file=sys.stderr)
        print(f"  time in draft calls   : {t_draft:.2f}s", file=sys.stderr)
        print(f"  time in verify calls  : {t_verify:.2f}s", file=sys.stderr)

    print("-" * 60, file=sys.stderr)
    print(f"draft model id          : {draft_id}", file=sys.stderr)
    print(f"target model id         : {target_id}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-url", required=True)
    p.add_argument("--target-model", required=True)
    p.add_argument("--draft-url", required=True)
    p.add_argument("--draft-model", required=True)
    p.add_argument("--tokenizer", default=None,
                   help="HF tokenizer name (defaults to --target-model)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--chat", action="store_true",
                   help="Wrap the prompt with the model's chat template")
    p.add_argument("--k", type=int, default=5,
                   help="Draft tokens proposed per round")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--stop-on-special", action="store_true",
                   help="Stop on ANY special token, not just eos_token_id")
    p.add_argument("--verbose", action="store_true",
                   help="Log each round to stderr: drafted tokens, "
                        "acceptance count, and the correction/bonus token")
    args = p.parse_args()
    asyncio.run(generate(args))


if __name__ == "__main__":
    main()
