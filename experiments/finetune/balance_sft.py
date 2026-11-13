"""Keep every answer example and half as many abstention examples (fixed seed).

    python experiments/finetune/balance_sft.py --input data/sft_train.jsonl \
        --out data/sft_train_balanced.jsonl

make_sft_data.py makes almost as many abstentions as answers (each question is also asked
against another document). Training on that mix teaches a small model to refuse too often, so
the LoRA run uses all answers and a random half-sized sample of the abstentions.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SEED = 20261109


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).parent
    source = args.input if args.input.is_absolute() else here / args.input
    target = args.out if args.out.is_absolute() else here / args.out
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    answers = [r for r in rows if r["kind"] == "answer"]
    abstentions = [r for r in rows if r["kind"] == "abstain"]
    random.Random(SEED).shuffle(abstentions)  # noqa: S311 - sampling, not security
    keep = answers + abstentions[: len(answers) // 2]
    random.Random(SEED).shuffle(keep)  # noqa: S311
    target.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep),
        encoding="utf-8",
        newline="\n",
    )
    print(len(keep), "examples:", len(answers), "answers,", len(keep) - len(answers), "abstain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
