import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


LETTERS = "ABCDEFGHIJ"

PATTERNS = [
    re.compile(r"the answer is\s*\(?\s*([A-J])\s*\)?", re.IGNORECASE),
    re.compile(r"answer is\s*\(?\s*([A-J])\s*\)?", re.IGNORECASE),
    re.compile(r"final answer\s*[:\-]?\s*\(?\s*([A-J])\s*\)?", re.IGNORECASE),
    re.compile(r"\b([A-J])\b", re.IGNORECASE),
    re.compile(r"([A-J])(?=[^A-J]*$)", re.IGNORECASE | re.DOTALL),  # last A-J
]

def extract_choice(text: str):
    t = (text or "").strip()
    if not t:
        return None
    for p in PATTERNS:
        m = p.search(t)
        if not m:
            continue
        cand = m.group(1) if (m.lastindex and m.lastindex >= 1) else m.group(0)
        cand = (cand or "").strip().upper()
        if cand and cand[0] in LETTERS:
            return cand[0]
    return None


def format_options(options):
    out = []
    for L, opt in zip(LETTERS, options):
        out.append(f"{L}) {opt}")
    return "\n".join(out)


def get_subject(ex: dict) -> str:
    for k in ("category", "subject", "topic", "domain"):
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "unknown"


def stderr(p: float, n: int) -> float:
    if n <= 0:
        return 0.0
    return math.sqrt(p * (1 - p) / n)


def best_prompt(question: str, options, subject: str) -> str:
    """
    Strong MMLU-style prompt:
    - subject priming
    - brief reasoning allowed
    - strict final format: 'the answer is (X)'
    - ends with 'the answer is (' to make completion easy
    """
    return (
        "You are taking a multiple-choice exam.\n"
        f"The following question is about: {subject}.\n\n"
        "Rules:\n"
        "- Use elimination and check for traps.\n"
        "- Keep reasoning brief.\n"
        "- You MUST finish with exactly this format:\n"
        "  the answer is (X)\n"
        "- X must be one of A,B,C,D,E,F,G,H,I,J.\n"
        "- Do not output anything after that final line.\n\n"
        f"Question: {question}\n\n"
        "Options:\n"
        f"{format_options(options)}\n\n"
        "the answer is ("
    )


def maybe_apply_chat_template(model_name: str, user_prompt: str, enable: bool):
    if not enable or AutoTokenizer is None:
        return user_prompt
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        if not getattr(tok, "chat_template", None):
            return user_prompt
        messages = [{"role": "user", "content": user_prompt}]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return user_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="swiss-ai/Apertus-8B-2509")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=8192)

    ap.add_argument("--limit", type=int, default=200, help="0 = full test")
    ap.add_argument("--out_jsonl", default="/workspace/mmlu/mmlu_pro_bestprompt_preds.jsonl")

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=96)

    ap.add_argument("--no_chat_template", action="store_true")
    args = ap.parse_args()

    # Load dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    # Load model
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    total = 0
    per_subj = defaultdict(lambda: [0, 0])

    with out_path.open("w", encoding="utf-8") as f:
        pbar = tqdm(ds, desc="MMLU-Pro (best prompt)", dynamic_ncols=True)
        for ex in pbar:
            q = ex.get("question") or ""
            options = ex.get("options") or []
            gold = ex.get("answer")

            if not q or not options or gold is None:
                continue

            subject = get_subject(ex)

            gold_letter = str(gold).strip().upper()
            if gold_letter.isdigit():
                gi = int(gold_letter)
                gold_letter = LETTERS[gi] if 0 <= gi < 10 else "A"
            if gold_letter not in LETTERS:
                gold_letter = "A"

            prompt = best_prompt(q, options, subject)
            prompt2 = maybe_apply_chat_template(args.model, prompt, enable=(not args.no_chat_template))

            raw = llm.generate([prompt2], sp)[0].outputs[0].text

            pred = extract_choice(raw) or "A"

            ok = (pred == gold_letter)
            correct += int(ok)
            total += 1
            per_subj[subject][0] += int(ok)
            per_subj[subject][1] += 1

            f.write(json.dumps({
                "subject": subject,
                "gold": gold_letter,
                "pred": pred,
                "correct": ok,
                "raw": raw,
            }, ensure_ascii=False) + "\n")

            if total:
                pbar.set_postfix_str(f"acc={correct/total:.3f} ({correct}/{total})")

    acc = correct / total if total else 0.0
    print("\n===== SUMMARY =====")
    print(f"Accuracy: {acc:.4f} ({correct}/{total})   StdErr: {stderr(acc, total):.4f}")
    print("Predictions JSONL:", str(out_path))

    print("\nPer-subject:")
    for subj in sorted(per_subj.keys()):
        c, n = per_subj[subj]
        a = c / n if n else 0.0
        print(f"{subj:20s}  {a:.4f} ± {stderr(a,n):.4f}   ({c}/{n})")


if __name__ == "__main__":
    main()

"""
python /workspace/mmlu/mmlu_pro_bestprompt_vllm.py \
  --model swiss-ai/Apertus-8B-2509 \
  --limit 0 \
  --max_tokens 96 \
  --out_jsonl /workspace/mmlu/out_bestprompt_full.jsonl
"""
