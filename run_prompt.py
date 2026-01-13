import argparse
import json
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_eval import evaluator
from lm_eval.api.model import LM
from lm_eval.utils import make_table


# Normalize stop sequences into a list of strings
def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


class PromptEvalModel(LM):
    """
    lm-eval wrapper for MMLU with prompt engineering only:
    - Wraps the lm-eval prompt with a custom instruction template
    - Uses HF model to compute loglikelihood for multiple-choice scoring
    """

    def __init__(
        self,
        pretrained: str,
        prompt_preamble: str,
        max_model_len: int = 8192,
    ):
        super().__init__()

        self.prompt_preamble = prompt_preamble

        # Load model and tokenizer for scoring
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=True, use_fast=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
        )
        self.model.eval()

        self.max_model_len = int(max_model_len)

    @property
    def eot_token_id(self):
        return None

    @property
    def max_length(self):
        return self.max_model_len

    def tok_encode(self, string):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    # Build the final prompt with the preamble
    def _build_prompt(self, prompt: str) -> str:
        return f"{self.prompt_preamble}\n\n{prompt}"

    # Loglikelihood for a single (prompt, continuation) pair
    def _loglikelihood_one(self, prompt: str, continuation: str) -> Tuple[float, bool]:
        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        continuation_ids = self.tokenizer(
            continuation, add_special_tokens=False, return_tensors="pt"
        ).input_ids

        input_ids = torch.cat([prompt_ids, continuation_ids], dim=1).to(self.model.device)
        with torch.no_grad():
            logits = self.model(input_ids).logits
        log_probs = torch.log_softmax(logits, dim=-1)

        prompt_len = prompt_ids.shape[1]
        cont_len = continuation_ids.shape[1]

        total_logprob = 0.0
        is_greedy = True
        for i in range(cont_len):
            tok_id = continuation_ids[0, i].item()
            lp = log_probs[0, prompt_len + i - 1, tok_id].item()
            total_logprob += lp
            if tok_id != int(torch.argmax(log_probs[0, prompt_len + i - 1]).item()):
                is_greedy = False

        return total_logprob, is_greedy

    # Lm-eval scoring path used by MMLU (multiple choice)
    def loglikelihood(self, requests):
        results = []
        for r in requests:
            context = r.args[0] if len(r.args) > 0 else ""
            continuation = r.args[1] if len(r.args) > 1 else ""

            full_prompt = self._build_prompt(str(context))
            results.append(self._loglikelihood_one(full_prompt, str(continuation)))

        return results

    # Generic generation path for tasks using generate_until
    def generate_until(self, requests):
        outputs = []
        for r in requests:
            prompt_text = r.args[0] if len(r.args) > 0 else ""
            stop_sequences = r.args[1] if len(r.args) > 1 else []
            stop = ensure_list(stop_sequences)

            full_prompt = self._build_prompt(str(prompt_text))
            input_ids = self.tokenizer(
                full_prompt, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.model.device)

            with torch.no_grad():
                out = self.model.generate(
                    input_ids,
                    max_new_tokens=32,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated_text = self.tokenizer.decode(
                out[0][input_ids.shape[1]:], skip_special_tokens=True
            )
            for s in stop:
                if s and s in generated_text:
                    generated_text = generated_text.split(s, 1)[0]
            outputs.append(generated_text)

        return outputs

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", default="swiss-ai/Apertus-8B-Instruct-2509")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--num_fewshot", type=int, default=5)
    ap.add_argument("--tasks", default="mmlu_stem,mmlu_humanities,mmlu_social_sciences,mmlu_other")
    ap.add_argument("--batch_size", default="auto")
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--log_samples", action="store_true")

    ap.add_argument(
        "--prompt_preamble",
        type=str,
        default=(
            "You are taking a multiple-choice exam. "
            "Identify the core concept, eliminate inconsistent options, "
            "and choose the best answer."
        ),
    )
    ap.add_argument("--max_model_len", type=int, default=8192)

    args = ap.parse_args()

    model = PromptEvalModel(
        pretrained=args.pretrained,
        prompt_preamble=args.prompt_preamble,
        max_model_len=args.max_model_len,
    )

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    res = evaluator.simple_evaluate(
        model=model,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=None if args.limit == 0 else args.limit,
        log_samples=args.log_samples,
    )

    out = {
        "results": res.get("results", {}),
        "group_results": res.get("group_results", {}),
        "config": {
            "pretrained": args.pretrained,
            "limit": args.limit,
            "num_fewshot": args.num_fewshot,
            "tasks": tasks,
            "batch_size": args.batch_size,
            "prompt_preamble": args.prompt_preamble,
            "max_model_len": args.max_model_len,
        },
    }

    outp = Path(args.output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(make_table(res, column="results"))
    if "group_results" in res:
        print(make_table(res, column="group_results"))

    print("Results written to:", outp)


if __name__ == "__main__":
    main()
