"""Validate, audit and score the groundedai-reliability-v1 suite.

    python scripts/reliability_suite.py validate [--root DIR]
    python scripts/reliability_suite.py rehash [--root DIR]
    python scripts/reliability_suite.py replace-cases --cases FILE.jsonl --reason TEXT [--root DIR]
    python scripts/reliability_suite.py audit-prepare --out DIR [--root DIR] [--seed N] [--round N]
        [--exclude IDS.json] [--include IDS.json]
    python scripts/reliability_suite.py audit-compare --sample FILE --answers FILE --out DIR
        [--root DIR]
    python scripts/reliability_suite.py audit-decide --result FILE --decisions FILE
    python scripts/reliability_suite.py score --split dev|test --checks v1|v2 [--root DIR]
        [--out DIR] [--paired-with SEALED.json] [--rerun-reason TEXT]

Nothing printed here contains the text of a held-out (test) case. Validation names case ids and
fields; scores are counts, rates and intervals. The blind audit page holds no labels, subtypes,
rationales or case ids.

Scoring the test split needs the frozen suite (its hash pinned in
``groundedai/evaluation/reliability.py``) and is allowed once per version of the checks; every
run is logged.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from groundedai.evaluation import reliability as suite  # noqa: E402
from groundedai.evaluation.environment import code_version, environment_info  # noqa: E402
from groundedai.evaluation.reliability import AUDIT_LABELS, Case  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "evaluation" / "datasets" / suite.SUITE_NAME
DEFAULT_OUT = REPO_ROOT / "evaluation" / "reports" / "reliability"
LOCAL_OUT = REPO_ROOT / "evaluation" / "reports" / "local" / "reliability"
AUDIT_SAMPLE = {"A": 14, "B": 9, "C": 12, "D": 15}
AUDIT_SEED = 20261015

CATEGORY_RULES = {
    "A": [
        "Judge only from the passages, never from outside knowledge.",
        "Supported: one passage states every factual part of the statement (a compound statement "
        "is supported if each part is stated in some passage).",
        'Unit conversions, number words and formats count as the same value ("two weeks" = '
        '"14 days", "8 am" = "08:00"); other arithmetic does not.',
        'Limit words carry meaning: "minimum 14" = "at least 14"; "at least 14" is not "at most '
        '14" or "exactly 14".',
        "Any unsupported part makes the whole statement not supported. Ignore whether the "
        "citation label is right; judge only whether the passages state it.",
    ],
    "B": [
        "Every statement in this group is true according to the passages; judge only the "
        "citation labels written in it.",
        "Correct: every cited label points to a passage that states part of the statement, and "
        "the cited passages together state all of it.",
        "Wrong: a cited passage does not state it, one of two cited passages states nothing of "
        "it, or a cited label is not among the passages.",
    ],
    "C": [
        "Judge only from the passages.",
        "An answer addresses the question when it states the asked item: the value or person "
        "for value questions, a clear yes or no for yes/no questions, the steps or the reason "
        "for how and why questions.",
        "Off target: the passages do contain the asked item, but the answer gives only other "
        "facts.",
        "Not answerable but answered: the passages do not contain the asked item, and the answer "
        "gives related facts as if answering.",
        "Correct refusal: the passages do not contain it, and the answer clearly says so.",
    ],
    "D": [
        "A disagreement is two passages from different documents that give incompatible "
        "information about the same thing, for the same group and period.",
        'Different stated periods ("from 1 March 2026", "2025 price list") -> different '
        "values for different periods.",
        'Only edit dates ("last updated"), or a date on one side only -> conflict.',
        "Different groups or plans, the same value written differently, a value equal to one "
        "end of a range, or a topic with no value on one side -> no disagreement.",
    ],
}


# ------------------------------------------------------------------------ helpers
def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _manifest(root: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return manifest


def _case_html(case: Case, number: int, *, reveal: dict[str, str] | None = None) -> str:
    passages = "".join(
        f'<div class="passage"><div class="plabel">[{html.escape(p.label)}] '
        f"{html.escape(p.title)}</div><div>{html.escape(p.text)}</div></div>"
        for p in case.passages
    )
    if case.category in "AB":
        shown = (
            f'<div class="block"><b>Statement:</b> {html.escape(str(case.data["statement"]))}</div>'
        )
    elif case.category == "C":
        shown = f'<div class="block"><b>Answer:</b> {html.escape(str(case.data["answer"]))}</div>'
    else:
        shown = ""
    name = f"case{number}"
    if reveal is None:
        options = "".join(
            f'<label class="opt"><input type="radio" name="{name}" value="{value}"> '
            f"{html.escape(text)}</label>"
            for value, text in AUDIT_LABELS[case.category]
        )
        form = (
            f'<div class="form">{options}<textarea name="{name}-note" '
            f'placeholder="Note (optional)"></textarea></div>'
        )
    else:
        labels = dict(AUDIT_LABELS[case.category])
        form = (
            f'<div class="form"><p><b>Your blind label:</b> '
            f"{html.escape(labels.get(reveal['blind'], reveal['blind']))}</p>"
            "<p><b>Proposed label:</b> "
            f"{html.escape(labels.get(reveal['gold'], reveal['gold']))}</p>"
            f"<p><b>Why (from the case writer):</b> {html.escape(reveal['rationale'])}</p>"
            f'<label class="opt"><input type="radio" name="{name}" value="keep"> Keep the proposed '
            f"label (my blind label was wrong)</label>"
            f'<label class="opt"><input type="radio" name="{name}" value="replace"> Replace this '
            f"case (the proposed label is wrong, or more than one label is defensible)</label>"
            f'<textarea name="{name}-note" placeholder="Note (optional)"></textarea></div>'
        )
    return (
        f'<section class="case" data-n="{number}"><h3>Case {number}</h3>'
        f'<div class="block"><b>Question:</b> {html.escape(case.question)}</div>'
        f"{passages}{shown}{form}</section>"
    )


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:,">
<title>{title}</title>
<style>
:root {{
  --bg:#fbfbf9; --fg:#1d1d1b; --muted:#5d5d58; --line:#d9d8d2; --card:#ffffff; --accent:#2f5d8a;
  --warn:#b3261e;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg:#18191a; --fg:#ececea; --muted:#a9a9a3; --line:#3a3b3d; --card:#222325;
    --accent:#8db3db; --warn:#f2b8b5;
  }}
}}
.case.missing {{ border:2px solid var(--warn); }}
#status.missing {{ color:var(--warn); font-weight:600; }}
body {{
  background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, sans-serif;
  margin:0; padding:16px;
}}
main {{ max-width:880px; margin:0 auto; }}
.case, .rules, .intro {{
  background:var(--card); border:1px solid var(--line); border-radius:8px;
  padding:14px 16px; margin:14px 0;
}}
.passage {{ border-left:3px solid var(--line); padding:4px 10px; margin:8px 0; font-size:15px; }}
.plabel {{ color:var(--muted); font-size:13px; }}
.block {{ margin:8px 0; }}
.opt {{ display:block; margin:4px 0; }}
textarea {{
  width:100%; min-height:40px; margin-top:6px; background:var(--bg); color:var(--fg);
  border:1px solid var(--line); border-radius:4px;
}}
button {{
  background:var(--accent); color:var(--bg); border:0; border-radius:6px;
  padding:10px 16px; font-size:16px; cursor:pointer;
}}
.bar {{
  position:sticky; bottom:0; background:var(--bg); padding:10px 0;
  border-top:1px solid var(--line);
}}
h2 {{ margin-top:28px; }}
</style></head><body><main>
<h1>{title}</h1>
<div class="intro">{intro}</div>
{body}
<div class="bar"><button id="save">{button}</button> <span id="status"></span></div>
</main>
<script>
const KEY = {key};
const FILE = {file};
const form = document.querySelector("main");
function collect() {{
  const out = {{}};
  document.querySelectorAll("section.case").forEach((section) => {{
    const picked = section.querySelector("input[type=radio]:checked");
    const note = section.querySelector("textarea").value;
    out[section.dataset.n] = {{ label: picked ? picked.value : null, note: note }};
  }});
  return out;
}}
function remember() {{
  try {{
    localStorage.setItem(KEY, JSON.stringify(collect()));
  }} catch (error) {{
    /* storage unavailable: the page still works, choices are not kept on reload */
  }}
}}
function restore() {{
  let saved = null;
  try {{
    saved = JSON.parse(localStorage.getItem(KEY) || "null");
  }} catch (error) {{
    saved = null;
  }}
  if (!saved) return;
  Object.entries(saved).forEach(([n, value]) => {{
    const section = document.querySelector('section.case[data-n="' + n + '"]');
    if (!section) return;
    if (value.label) {{
      const input = section.querySelector('input[value="' + value.label + '"]');
      if (input) input.checked = true;
    }}
    section.querySelector("textarea").value = value.note || "";
  }});
}}
form.addEventListener("change", remember);
form.addEventListener("input", remember);
document.getElementById("save").addEventListener("click", () => {{
  const data = collect();
  const missing = Object.entries(data).filter(([, value]) => !value.label).map(([n]) => n);
  document.querySelectorAll("section.case").forEach((section) => {{
    section.classList.toggle("missing", missing.includes(section.dataset.n));
  }});
  const blob = new Blob([JSON.stringify(data, null, 2)], {{ type: "application/json" }});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = FILE;
  link.click();
  const status = document.getElementById("status");
  status.classList.toggle("missing", missing.length > 0);
  const several = missing.length > 1;
  status.textContent = missing.length
    ? "Saved, but " + (several ? "cases " : "case ") + missing.join(", ") + " still " +
      (several ? "have" : "has") + " no choice (marked in red). Choose, then save again."
    : "Saved as " + FILE + ".";
}});
restore();
</script></body></html>
"""

CATEGORY_TITLES = {
    "A": "Group A: is the statement supported by the passages?",
    "B": "Group B: are the citation labels right?",
    "C": "Group C: does the answer address the question?",
    "D": "Group D: do the passages disagree?",
}


def _page(
    title: str, intro: str, sections: dict[str, list[str]], *, key: str, file: str, button: str
) -> str:
    body = ""
    for category in suite.CATEGORIES:
        if not sections.get(category):
            continue
        rules = "".join(f"<li>{html.escape(rule)}</li>" for rule in CATEGORY_RULES[category])
        body += (
            f"<h2>{html.escape(CATEGORY_TITLES[category])}</h2>"
            f'<div class="rules"><b>Rules</b><ul>{rules}</ul></div>' + "".join(sections[category])
        )
    return PAGE.format(
        title=html.escape(title),
        intro=intro,
        body=body,
        key=json.dumps(key),
        file=json.dumps(file),
        button=html.escape(button),
    )


# ----------------------------------------------------------------------- commands
def cmd_validate(args: argparse.Namespace) -> int:
    issues, counts = suite.validate_suite(Path(args.root))
    for split, categories in counts.items():
        total = sum(sum(c.values()) for c in categories.values())
        per = ", ".join(f"{k} {sum(v.values())}" for k, v in categories.items())
        print(f"{split}: {total} cases ({per})")
    errors = [issue for issue in issues if issue.severity == "error"]
    warnings = [issue for issue in issues if issue.severity == "warning"]
    for issue in errors + warnings[: args.max_warnings]:
        print(issue)
    if len(warnings) > args.max_warnings:
        print(f"... {len(warnings) - args.max_warnings} more warnings")
    print(f"{len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


def cmd_rehash(args: argparse.Namespace) -> int:
    root = Path(args.root)
    manifest = _manifest(root)
    manifest.update(suite.hash_record(root))
    _write(root / "manifest.json", suite.as_json(manifest))
    print(f"files {len(manifest['files'])}; test_sha256 {manifest['test_sha256']}")
    print(f"dev_sha256 {manifest['dev_sha256']}; combined_sha256 {manifest['combined_sha256']}")
    return 0


def cmd_replace_cases(args: argparse.Namespace) -> int:
    """Put replacement cases in place of existing ones, keeping id, category, split and subtype.

    Every replacement is logged in ``REVISIONS.jsonl`` (next to the manifest) with the old and new
    line hashes and the reason; case text is never printed or logged. Run ``rehash`` and
    ``validate`` afterwards; a replaced test case must then be audited blind.
    """
    root = Path(args.root)
    by_file: dict[Path, dict[str, dict[str, Any]]] = {}
    for line in Path(args.cases).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or str(row.get("split")) not in suite.SPLITS:
            print("error: every replacement needs a split", file=sys.stderr)
            return 2
        path = root / str(row["split"]) / f"{row.get('category')}.jsonl"
        if not path.is_file():
            print(f"error: no case file for replacement {row.get('id')}", file=sys.stderr)
            return 2
        if str(row.get("id")) in by_file.setdefault(path, {}):
            print(f"error: two replacements for {row.get('id')}", file=sys.stderr)
            return 2
        by_file[path][str(row.get("id"))] = row
    # Check everything before writing anything, so a refused run changes no file.
    updated: dict[Path, str] = {}
    revisions: list[dict[str, Any]] = []
    today = datetime.now(UTC).date().isoformat()
    for path, replacements in by_file.items():
        lines = path.read_text(encoding="utf-8").splitlines()
        found: set[str] = set()
        for index, line in enumerate(lines):
            old = json.loads(line) if line.strip() else {}
            new = replacements.get(str(old.get("id")))
            if new is None:
                continue
            for key in ("category", "split", "subtype"):
                if new.get(key) != old.get(key):
                    print(f"error: {old['id']}: a replacement must keep its {key}", file=sys.stderr)
                    return 2
            text = json.dumps(new, ensure_ascii=False)
            lines[index] = text
            found.add(str(old["id"]))
            revisions.append(
                {
                    "id": old["id"],
                    "subtype": old["subtype"],
                    "old_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
                    "new_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "reason": args.reason,
                    "date": today,
                }
            )
        missing = sorted(set(replacements) - found)
        if missing:
            print(f"error: no existing case with id {missing}", file=sys.stderr)
            return 2
        updated[path] = "\n".join(lines) + "\n"
    for path, text in updated.items():
        _write(path, text)
    with (root / "REVISIONS.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        for entry in revisions:
            handle.write(json.dumps(entry) + "\n")
    print(f"replaced {len(revisions)} case(s): {sorted(r['id'] for r in revisions)}")
    return 0


def _id_list(path: str | None) -> list[str]:
    return [str(item) for item in json.loads(Path(path).read_text("utf-8"))] if path else []


def cmd_audit_prepare(args: argparse.Namespace) -> int:
    """Draw a blind audit sample and write its page.

    ``--exclude`` lists case ids never to sample (cases audited before). ``--include`` lists case
    ids that must be on the page besides the random sample (replacement cases, which are audited
    blind as well); they are shuffled in with the sampled cases of their group, and the sample
    file records which numbers they have, so the random sample is scored on its own.
    """
    root, out = Path(args.root), Path(args.out)
    cases = suite.load_cases(root, "test")
    by_id = {case.id: case for case in cases}
    include = _id_list(args.include)
    unknown = [case_id for case_id in include if case_id not in by_id]
    if unknown:
        print(f"error: --include names ids that are not test cases: {unknown}", file=sys.stderr)
        return 2
    exclude = set(_id_list(args.exclude)) | set(include)
    rng = random.Random(args.seed)  # noqa: S311 - reproducible sampling, not security
    chosen: list[Case] = []
    for category in suite.CATEGORIES:
        pool = sorted(
            (c for c in cases if c.category == category and c.id not in exclude), key=lambda c: c.id
        )
        picked = rng.sample(pool, AUDIT_SAMPLE[category])
        picked += [by_id[case_id] for case_id in include if by_id[case_id].category == category]
        rng.shuffle(picked)
        chosen.extend(picked)
    numbers = {case.id: index for index, case in enumerate(chosen, start=1)}
    sections: dict[str, list[str]] = {}
    for case in chosen:
        sections.setdefault(case.category, []).append(_case_html(case, numbers[case.id]))
    file = "blind_answers.json" if args.round == 1 else f"blind_answers_round{args.round}.json"
    intro = (
        "<p>Label each case yourself, using only the passages and the rules of its group. The "
        "page shows no proposed labels and no system output.</p><p>Your choices are kept in this "
        "browser while you work. When you have finished, press <b>Save my answers</b> at the "
        f"bottom: it downloads <code>{file}</code>. Send that file back.</p>"
    )
    title = "GroundedAI label audit (blind)"
    if args.round != 1:
        title = f"GroundedAI label audit, round {args.round} (blind)"
    page = _page(
        title,
        intro,
        sections,
        key=f"{suite.SUITE_NAME}-blind-{args.seed}",
        file=file,
        button="Save my answers",
    )
    _write(out / "blind_audit.html", page)
    sample: dict[str, Any] = {
        "seed": args.seed,
        "test_sha256": suite.hash_record(root)["test_sha256"],
        "numbers": {str(n): case_id for case_id, n in numbers.items()},
    }
    if args.round != 1 or include:
        sample["round"] = args.round
        sample["included"] = sorted((str(numbers[case_id]) for case_id in include), key=int)
    _write(out / "audit_sample.json", suite.as_json(sample))
    counts = {category: AUDIT_SAMPLE[category] for category in suite.CATEGORIES}
    print(
        f"audit page: {out / 'blind_audit.html'}; {len(chosen)} cases: random sample {counts}"
        + (f", plus {len(include)} included case(s)" if include else "")
    )
    return 0


def _answers(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): v if isinstance(v, dict) else {"label": v} for k, v in data.items()}


class AnswersError(ValueError):
    """An answers file that cannot be matched to the audit sample."""


def check_answers(path: Path, categories: dict[str, str]) -> tuple[dict[str, str], dict[str, Any]]:
    """Read a saved answers file and match it to the audit sample.

    ``categories`` maps each audit number to its case's category. Returns the valid labels by
    number and a summary naming audit numbers only (never a label or a note). Duplicate keys and
    keys that are not audit numbers make the file incompatible: it is then refused, because it is
    unclear which answer is meant. A missing or empty label, or one that is not a label of the
    case's group, is reported and counts as not confirming the proposed label.
    """
    duplicates: list[str] = []

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: dict[str, Any] = {}
        for key, value in pairs:
            if key in seen:
                duplicates.append(key)
            seen[key] = value
        return seen

    raw = path.read_bytes()
    try:
        data = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnswersError(f"{path.name} is not readable JSON ({exc.__class__.__name__})") from exc
    if not isinstance(data, dict):
        raise AnswersError(f"{path.name} must hold one object keyed by audit number")
    unmatched = sorted((key for key in data if key not in categories), key=str)
    if duplicates or unmatched:
        raise AnswersError(
            f"{path.name} does not match the audit sample: duplicate keys "
            f"{sorted(set(duplicates))}, keys that are not audit numbers {unmatched}"
        )
    labels: dict[str, str] = {}
    missing: list[str] = []
    invalid: list[str] = []
    malformed: list[str] = []
    for number in sorted(categories, key=int):
        value = data.get(number)
        if isinstance(value, dict):
            label = value.get("label")
        elif value is None or isinstance(value, str):
            label = value
        else:
            malformed.append(number)
            continue
        if label is None or (isinstance(label, str) and not label.strip()):
            missing.append(number)
        elif not isinstance(label, str) or label not in dict(AUDIT_LABELS[categories[number]]):
            invalid.append(number)
        else:
            labels[number] = label
    summary = {
        "file": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "entries": len(data),
        "answered": len(labels),
        "missing": missing,
        "invalid": invalid + malformed,
    }
    return labels, summary


def cmd_audit_compare(args: argparse.Namespace) -> int:
    root, out = Path(args.root), Path(args.out)
    sample = json.loads(Path(args.sample).read_text(encoding="utf-8"))
    cases = {case.id: case for case in suite.load_cases(root, "test")}
    if sample.get("test_sha256") != suite.hash_record(root)["test_sha256"]:
        print("error: the test split changed after the audit sample was drawn", file=sys.stderr)
        return 2
    numbers = sorted(sample["numbers"].items(), key=lambda item: int(item[0]))
    try:
        answers, checked = check_answers(
            Path(args.answers), {number: cases[case_id].category for number, case_id in numbers}
        )
    except AnswersError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"answers: {checked['entries']} entries for {len(numbers)} audit cases; "
        f"{checked['answered']} with a valid label; missing {checked['missing'] or 'none'}; "
        f"invalid {checked['invalid'] or 'none'}; no duplicate or unknown keys"
    )
    # Included (replacement) cases are scored apart from the random sample, which alone is
    # compared with the agreement threshold.
    included = {str(number) for number in sample.get("included") or []}
    per: dict[str, list[bool]] = {category: [] for category in suite.CATEGORIES}
    per_included: dict[str, list[bool]] = {category: [] for category in suite.CATEGORIES}
    disagreements: list[dict[str, Any]] = []
    for number, case_id in numbers:
        case = cases[case_id]
        blind = answers.get(number)
        agree = blind == suite.gold_label(case)
        (per_included if number in included else per)[case.category].append(agree)
        if not agree:
            disagreements.append(
                {
                    "number": number,
                    "id": case_id,
                    "blind": blind or "none",
                    "reason": "no valid blind label" if blind is None else "different label",
                    "included": number in included,
                }
            )

    def agreement(name: str, groups: dict[str, list[bool]]) -> tuple[int, int]:
        total = sum(len(v) for v in groups.values())
        agreed = sum(sum(v) for v in groups.values())
        interval = suite.wilson(agreed, total)
        share = ""
        if interval:
            share = f" ({agreed / total:.0%}; 95% CI {interval[0]:.2f}-{interval[1]:.2f})"
        print(f"{name}: {agreed} of {total}{share}")
        for category, values in groups.items():
            if values or not included:
                print(f"  {category}: {sum(values)} of {len(values)}")
        return agreed, total

    agreed, total = agreement(
        "blind agreement, random sample" if included else "blind agreement", per
    )
    if included:
        agreement("included (replacement) cases, each of which must agree", per_included)
    unconfirmed = checked["missing"] + checked["invalid"]
    if unconfirmed:
        labelled = len(numbers) - len(unconfirmed)
        agreed_all = len(numbers) - len(disagreements)
        print(
            f"  {len(unconfirmed)} case(s) without a valid blind label count as not confirmed "
            f"(audit numbers {unconfirmed}); among the {labelled} labelled cases, "
            f"{agreed_all} agree"
        )
    sections: dict[str, list[str]] = {}
    for item in disagreements:
        case = cases[item["id"]]
        reveal = {
            "blind": item["blind"],
            "gold": suite.gold_label(case),
            "rationale": str(case.data.get("rationale", "")),
        }
        sections.setdefault(case.category, []).append(
            _case_html(case, int(item["number"]), reveal=reveal)
        )
    if disagreements:
        intro = (
            "<p>These are the cases where your blind label differs from the proposed label. For "
            "each one, decide whether to keep the proposed label or to replace the case. Then "
            "press <b>Save my decisions</b>: it downloads <code>audit_decisions.json</code>.</p>"
        )
        _write(
            out / "reveal.html",
            _page(
                "GroundedAI label audit: disagreements",
                intro,
                sections,
                key=f"{suite.SUITE_NAME}-reveal-{sample['seed']}",
                file="audit_decisions.json",
                button="Save my decisions",
            ),
        )
        print(f"reveal page for {len(disagreements)} disagreement(s): {out / 'reveal.html'}")
    result: dict[str, Any] = {
        "seed": sample["seed"],
        "test_sha256": sample["test_sha256"],
        "answers": checked,
        "agreed": agreed,
        "total": total,
        "per_category": {c: [sum(v), len(v)] for c, v in per.items()},
        "disagreements": [
            {"number": d["number"], "id": d["id"], "reason": d["reason"]}
            | ({"included": d["included"]} if included else {})
            for d in disagreements
        ],
    }
    if included:
        result["included"] = {
            "agreed": sum(sum(v) for v in per_included.values()),
            "total": sum(len(v) for v in per_included.values()),
            "per_category": {c: [sum(v), len(v)] for c, v in per_included.items() if v},
        }
    _write(out / "audit_result.json", suite.as_json(result))
    return 0


def cmd_audit_decide(args: argparse.Namespace) -> int:
    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    decisions = _answers(Path(args.decisions))
    keep: list[str] = []
    replace: list[str] = []
    missing: list[str] = []
    for item in result["disagreements"]:
        choice = (decisions.get(str(item["number"])) or {}).get("label")
        (keep if choice == "keep" else replace if choice == "replace" else missing).append(
            item["id"]
        )
    print(f"keep {len(keep)}, replace {len(replace)}, undecided {len(missing)}")
    _write(
        Path(args.result).with_name("audit_decisions_applied.json"),
        suite.as_json({"keep": keep, "replace": replace, "undecided": missing}),
    )
    return 1 if missing else 0


def _row(*cells: object) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {suite.SUITE_NAME}: {report['split']} split, checks {report['checks']}",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Code: {report['code']}",
        f"- Check versions: {report['versions']}",
        f"- Dataset {report['split']} hash: `{report['dataset']['split_sha256']}`",
        f"- Cases: {report['dataset']['cases']}",
        "",
        "Rates with 95% Wilson intervals. *Met*: the point estimate reaches the target; "
        "*established*: the interval's bound does too.",
        "",
        "| Measure | k / n | Value | 95% CI | Target (stretch) | Met | Stretch met | Established |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for category, measures in report["summary"].items():
        for name, record in measures.items():
            if not isinstance(record, dict):
                lines.append(
                    f"| {category}. {name} | | {record if record is not None else '–'} | | | | | |"
                )
                continue
            ci = f"{record['ci95'][0]:.3f}–{record['ci95'][1]:.3f}" if record.get("ci95") else "–"
            value = f"{record['value']:.3f}" if record.get("value") is not None else "–"
            target = (
                f"{'≥' if record['direction'] == 'higher' else '≤'} {record['target']} "
                f"({record['stretch']})"
                if "target" in record
                else ""
            )
            flags = [
                ("yes" if record[key] else "no") if key in record else ""
                for key in ("met", "stretch_met", "established")
            ]
            lines.append(
                _row(
                    f"{category}. {name}",
                    f"{record['k']} / {record['n']}",
                    value,
                    ci,
                    target,
                    *flags,
                )
            )
    lines += ["", "## Correctly handled cases per subtype (descriptive)", ""]
    lines += ["| Category | Subtype | k / n | Rate | 95% CI |", "|---|---|---|---|---|"]
    for category, subtypes in report["by_subtype"].items():
        for subtype, record in subtypes.items():
            ci = f"{record['ci95'][0]:.2f}–{record['ci95'][1]:.2f}" if record.get("ci95") else "–"
            rate = f"{record['value']:.2f}"
            lines.append(_row(category, subtype, f"{record['k']} / {record['n']}", rate, ci))
    if report.get("paired"):
        lines += ["", "## Paired comparison with the baseline (exact McNemar)", ""]
        lines += ["| Category | Cases | Fixed | Broke | p |", "|---|---|---|---|---|"]
        for category, record in report["paired"]["by_category"].items():
            lines.append(
                _row(category, record["cases"], record["fixed"], record["broke"], record["p_value"])
            )
        lines.append(f"\nBaseline run: `{report['paired']['baseline']}`")
    lines += ["", "## Limitations", ""] + [f"- {item}" for item in report["limitations"]]
    return "\n".join(lines) + "\n"


LIMITATIONS = [
    "Synthetic cases written by a separate agent of the same model family as the checks' author; "
    "shared blind spots are possible. A blind audit of 50 test labels estimates label quality.",
    "Subtypes were chosen around known failure modes, so the mix does not reflect how often each "
    "case occurs in real use; overall rates are not real-world rates.",
    "The checks are lexical and pattern-based. A good score does not show that they understand "
    "meaning, and results on this suite do not prove generalisation to other documents.",
    "Component-level only: fixed passages and fixed answers; no language model was run.",
]


def cmd_score(args: argparse.Namespace) -> int:
    root, out = Path(args.root), Path(args.out)
    checks = suite.CHECKS[args.checks]
    manifest = _manifest(root)
    record = suite.hash_record(root)
    split_hash = record[f"{args.split}_sha256"]
    log_path = out / "test-runs.jsonl"
    if args.split == "test":
        pinned = suite.FROZEN_TEST_SHA256
        if manifest.get("status") != "frozen" or pinned is None:
            print(
                "error: the test split is not frozen; score the dev split instead", file=sys.stderr
            )
            return 2
        if split_hash != pinned or manifest.get("test_sha256") != pinned:
            print(
                "error: the test split differs from the frozen version; refusing to score it",
                file=sys.stderr,
            )
            return 2
        previous = [
            json.loads(line)
            for line in (log_path.read_text("utf-8").splitlines() if log_path.is_file() else [])
            if line.strip()
        ]
        if any(run["checks"] == checks.name for run in previous) and not args.rerun_reason:
            print(
                f"error: the test split was already scored with checks {checks.name}; it is "
                "scored once per version (pass --rerun-reason to override, which is logged)",
                file=sys.stderr,
            )
            return 2
    cases = suite.load_cases(root, args.split)
    predictions, outcomes = suite.run_checks(cases, checks)
    generated = datetime.now(UTC)
    report_id = f"{args.split}-{checks.name}-{generated:%Y%m%dT%H%M%SZ}"
    report: dict[str, Any] = {
        "report_id": report_id,
        "kind": "reliability",
        "suite": suite.SUITE_NAME,
        "split": args.split,
        "checks": checks.name,
        "versions": dict(checks.versions),
        "scoring_version": suite.SCORING_VERSION,
        "generated_at": generated.isoformat(timespec="seconds"),
        "code": code_version(),
        "environment": {k: v for k, v in environment_info().items() if k != "packages"},
        "dataset": {
            "split_sha256": split_hash,
            "combined_sha256": record["combined_sha256"],
            "cases": len(cases),
        },
        "summary": suite.summarize(list(outcomes.values())),
        "by_subtype": suite.by_subtype(list(outcomes.values())),
        "limitations": LIMITATIONS,
    }
    if args.paired_with:
        baseline = json.loads(Path(args.paired_with).read_text(encoding="utf-8"))
        report["paired"] = {
            "baseline": baseline["report_id"],
            "by_category": suite.paired(baseline["outcomes"], outcomes),
        }
    _write(out / f"{report_id}.json", suite.as_json(report))
    _write(out / f"{report_id}.md", _markdown(report))
    if args.split == "test":
        sealed = {
            "report_id": report_id,
            "outcomes": {
                case_id: {"category": o["category"], "correct": o["correct"]}
                for case_id, o in outcomes.items()
            },
        }
        _write(out / "sealed" / f"{report_id}.json", suite.as_json(sealed))
        entry = {
            "report_id": report_id,
            "checks": checks.name,
            "generated_at": report["generated_at"],
            "code": report["code"],
            "test_sha256": split_hash,
            "rerun_reason": args.rerun_reason,
        }
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry) + "\n")
    else:
        details = {
            case.id: {
                "subtype": case.subtype,
                "gold": dict(case.gold),
                "prediction": predictions[case.id],
                "outcome": outcomes[case.id],
            }
            for case in cases
        }
        _write(LOCAL_OUT / f"{report_id}-cases.json", suite.as_json(details))
        print(f"per-case details: {LOCAL_OUT / (report_id + '-cases.json')}")
    print(_markdown(report))
    print(f"written: {out / (report_id + '.json')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    root_help = "suite folder (default: the committed suite)"
    p = sub.add_parser("validate")
    p.add_argument("--root", default=str(DEFAULT_ROOT), help=root_help)
    p.add_argument("--max-warnings", type=int, default=20)
    p.set_defaults(func=cmd_validate)
    p = sub.add_parser("rehash")
    p.add_argument("--root", default=str(DEFAULT_ROOT), help=root_help)
    p.set_defaults(func=cmd_rehash)
    p = sub.add_parser("replace-cases")
    p.add_argument("--root", default=str(DEFAULT_ROOT), help=root_help)
    p.add_argument("--cases", required=True, help="JSONL file of replacement cases")
    p.add_argument("--reason", required=True)
    p.set_defaults(func=cmd_replace_cases)
    p = sub.add_parser("audit-prepare")
    p.add_argument("--root", default=str(DEFAULT_ROOT), help=root_help)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=AUDIT_SEED)
    p.add_argument("--round", type=int, default=1, help="audit round (names the answers file)")
    p.add_argument("--exclude", default=None, help="JSON list of case ids not to sample")
    p.add_argument(
        "--include", default=None, help="JSON list of case ids to add (replacement cases)"
    )
    p.set_defaults(func=cmd_audit_prepare)
    p = sub.add_parser("audit-compare")
    p.add_argument("--root", default=str(DEFAULT_ROOT), help=root_help)
    p.add_argument("--sample", required=True)
    p.add_argument("--answers", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_audit_compare)
    p = sub.add_parser("audit-decide")
    p.add_argument("--result", required=True)
    p.add_argument("--decisions", required=True)
    p.set_defaults(func=cmd_audit_decide)
    p = sub.add_parser("score")
    p.add_argument("--root", default=str(DEFAULT_ROOT), help=root_help)
    p.add_argument("--split", choices=suite.SPLITS, required=True)
    p.add_argument("--checks", choices=sorted(suite.CHECKS), required=True)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--paired-with", default=None, help="sealed outcomes of the baseline test run")
    p.add_argument("--rerun-reason", default=None)
    p.set_defaults(func=cmd_score)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
