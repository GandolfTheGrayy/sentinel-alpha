"""
Sentinel Daily Build Script

Reads TODO.md to determine today's priorities, then uses Claude to generate
1-11 modular commits that meaningfully advance the Sentinel Sentiment Engine.
"""
import json
import os
import random
import subprocess
import sys
import textwrap
from datetime import date
from pathlib import Path

import anthropic

COMMITS_PER_DAY = random.randint(1, 4)
STATE_PATH = Path("scripts/build_state.json")

PROJECT_CONTEXT = """
You are a senior engineer building the Sentinel Sentiment Engine — an autonomous
financial intelligence system that predicts stock price movements by analyzing
niche sentiment signals cross-referenced with historical market data via RAG.

ARCHITECTURE:
  sentinel/scout/     — Data ingestion: scrapers for live prices, SEC filings,
                        Reddit/HN sentiment, GitHub developer health signals
  sentinel/linguist/  — LLM reasoning: "certainty vs. hesitation" analysis,
                        Linguistic Drift detection, Regulatory Whispers detector
  sentinel/historian/ — RAG pipeline: ChromaDB vector DB, historical event
                        lookup, confidence score weighting
  sentinel/judge/     — Daily post-mortem: predicted vs. actual market moves,
                        heuristic refinement, anomaly flagging

LLM USAGE (strict):
  - Claude via `anthropic` SDK (model "claude-sonnet-4-6") — reasoning ONLY:
    Linguist analyses, Judge calibration, Historian RAG synthesis.
  - Gemini via `google-generativeai` SDK (model "gemini-3.1-flash-lite-preview") —
    web scraping, HTML/text extraction, high-volume parsing in Scout modules.
  - Never call Claude for scraping or bulk extraction. Never call Gemini for
    nuanced reasoning or final scoring.
  - Read keys from env: ANTHROPIC_API_KEY, GEMINI_API_KEY.

CODING RULES:
  1. Every file must have a module-level docstring explaining its role in Sentinel.
  2. Functions must have type hints and docstrings.
  3. Use only stdlib + these approved packages: anthropic, google-generativeai,
     yfinance, praw, chromadb, requests, beautifulsoup4, pyyaml, sqlite3,
     numpy, pandas.
  4. Each output must be a single complete, importable Python module or script.
  5. Output ONLY raw source code — no markdown fences, no prose, no preamble.
  6. On the very first line, output the target file path prefixed with:
       # path: sentinel/scout/_generated/sec_scraper.py
     Then a blank line, then the code. AI-generated modules ALWAYS go in
     the `_generated/` subdir of their pillar — never in the pillar root,
     which is reserved for hand-written spine modules.
"""

PILLAR_TASKS = {
    "scout": [
        "A modular yfinance-based live price fetcher that stores OHLCV data in SQLite with a swap-ready interface for TimescaleDB",
        "An SEC EDGAR RSS scraper that polls the 8-K and 10-Q feeds and extracts filing metadata into a normalized dataclass",
        "A Reddit sentiment scraper using PRAW targeting r/wallstreetbets, r/stocks, and r/investing, outputting a normalized SentimentSignal dataclass",
        "A Hacker News scraper targeting 'Ask HN' posts about tech companies, scoring developer community sentiment",
        "A GitHub repository health signal collector measuring stars, commit velocity (commits/week), and issue open rate for a given repo",
        "A data normalizer that maps outputs from all scrapers into a unified SignalRecord schema stored in SQLite",
        "A config loader that reads a YAML config file and environment variables, with a typed Settings dataclass",
        "A base time-series SQLite schema module — creates tables for price history, sentiment signals, and prediction records",
    ],
    "linguist": [
        "A prompt template system for LLM-based 'certainty vs. hesitation' scoring of corporate text, returning a structured CertaintyScore dataclass",
        "A Linguistic Drift detector that compares a company's current 10-Q language against a rolling 30-day baseline and flags significant tone shifts",
        "A Regulatory Whispers detector that scans SEC filings for hedging language patterns (e.g. 'may', 'subject to', 'could materially') and scores their density",
        "An earnings call transcript parser that segments text by speaker role (CEO, CFO, Analyst) and prepares each segment for LLM analysis",
        "A sentiment aggregator that combines Scout signals and Linguist scores into a composite SentimentResidual score with a weighted formula",
        "A 'tells' extractor — given a block of corporate text, uses Claude to identify specific linguistic tells that historically precede price moves",
    ],
    "historian": [
        "A ChromaDB vector database setup module — initializes the local DB, defines collections for market events and filings, and provides a typed client wrapper",
        "A historical market event ingestion pipeline that reads from a CSV of past events and embeds them into ChromaDB using a simple embedding function",
        "A RAG query interface — given a current SentimentResidual, queries ChromaDB for the top-k most similar historical events and returns a HistoricalMatch list",
        "A confidence score weighting system that combines RAG similarity scores with recency decay to produce a final WeightedConfidence float",
        "An event schema module defining dataclasses for MarketEvent, HistoricalMatch, and ConfidenceReport used across the Historian layer",
    ],
    "judge": [
        "A post-mortem report generator that reads yesterday's PredictionRecord from SQLite, fetches actual price data, and writes a markdown report to backtest_results/",
        "A Predicted Residual vs. Actual Market Move comparator that calculates directional accuracy and magnitude error, returning a CalibrationResult",
        "A heuristic update logger that appends CalibrationResult entries to a JSONL file and computes rolling 7-day and 30-day accuracy metrics",
        "An anomaly flagging system that detects when actual market moves exceed 2x the predicted residual and generates an AnomalyAlert dataclass",
        "A daily summary printer that reads the latest post-mortem and prints a concise console summary of Sentinel's performance",
    ],
    "tests": [
        "A pytest unit test module for the Scout price fetcher — mocks yfinance responses and asserts correct SQLite writes",
        "A pytest unit test module for the config loader — tests env var overrides, missing key handling, and type coercion",
        "A pytest unit test module for the Linguistic Drift detector — uses fixture text to assert correct drift scoring",
        "A pytest integration test that runs the Scout → Linguist pipeline end-to-end with mocked external calls",
    ],
}


def run(cmd: list[str]) -> str:
    """Run a shell command, capture output, and return stripped stdout."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def read_todo() -> str:
    """Return the contents of TODO.md, or a fallback message if missing."""
    todo_path = Path("TODO.md")
    if todo_path.exists():
        return todo_path.read_text(encoding="utf-8")
    return "No TODO.md found. Focus on base infrastructure."


def load_state() -> dict:
    """Load persisted build state (which (pillar, task) pairs are done)."""
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"completed": [], "iterations": {}}


def save_state(state: dict) -> None:
    """Persist build state to disk."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def pick_tasks(today: date, state: dict) -> list[tuple[str, str]]:
    """Pick today's tasks, preferring un-built ones; recycle once exhausted."""
    random.seed(today.toordinal())
    completed = {tuple(t) for t in state["completed"]}
    flat = [(p, t) for p, ts in PILLAR_TASKS.items() for t in ts]
    available = [pt for pt in flat if pt not in completed]
    if len(available) < COMMITS_PER_DAY:
        recycled = [pt for pt in flat if pt in completed]
        random.shuffle(recycled)
        available = available + recycled
    random.shuffle(available)
    return available[:COMMITS_PER_DAY]


def generate_module(
    client: anthropic.Anthropic,
    pillar: str,
    task: str,
    todo_context: str,
    index: int,
) -> tuple[str, str, str]:
    """Ask Claude to generate a Sentinel module. Returns (file_path, code, commit_msg)."""
    prompt = textwrap.dedent(f"""
        {PROJECT_CONTEXT}

        CURRENT TODO.MD CONTEXT:
        {todo_context[:1500]}

        TODAY'S TASK (commit {index} of {COMMITS_PER_DAY}):
        Pillar: {pillar}
        Build: {task}

        Requirements:
        - Output the file path on line 1 as:  # path: sentinel/{pillar}/_generated/filename.py
          (or sentinel/tests/_generated/filename.py for test modules)
        - Then a blank line, then the complete Python module.
        - The module must be self-contained and importable.
        - Include a module docstring explaining how this file fits into Sentinel.
        - Every public function needs type hints and a one-line docstring.
    """).strip()

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()

    lines = raw.splitlines()
    file_path = f"sentinel/{pillar}/_generated/module_{index:02d}.py"
    code_lines = lines
    if lines and lines[0].startswith("# path:"):
        file_path = lines[0].split(":", 1)[1].strip()
        code_lines = lines[2:] if len(lines) > 2 else lines[1:]

    p = Path(file_path)
    if "_generated" not in p.parts:
        parts = list(p.parts)
        if len(parts) >= 2 and parts[0] == "sentinel":
            parts.insert(2, "_generated")
            p = Path(*parts)
        else:
            p = Path("sentinel") / pillar / "_generated" / p.name
    file_path = str(p).replace("\\", "/")

    code = "\n".join(code_lines).strip() + "\n"
    filename = p.stem.replace("_", " ")
    commit_msg = f"feat({pillar}): {filename}"
    return file_path, code, commit_msg


def update_todo(completed: list[tuple[str, str]]) -> None:
    """Append today's completed tasks under the ## Completed section of TODO.md."""
    todo_path = Path("TODO.md")
    if not todo_path.exists():
        return
    content = todo_path.read_text(encoding="utf-8")
    today = date.today().isoformat()
    entry = f"\n### {today}\n" + "\n".join(f"- [x] {task[:80]}" for _, task in completed)
    marker = "## Completed (AI scaffolding"
    if marker in content:
        idx = content.index(marker)
        line_end = content.index("\n", idx) + 1
        content = content[:line_end] + entry + "\n" + content[line_end:]
    elif "## Completed" in content:
        content = content.replace("## Completed", f"## Completed (AI scaffolding — `_generated/` only, not production)\n{entry}")
    else:
        content += f"\n## Completed (AI scaffolding — `_generated/` only, not production)\n{entry}\n"
    todo_path.write_text(content, encoding="utf-8")


def main() -> None:
    """Entry point — generate today's batch of Sentinel commits."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)
    today = date.today()
    todo_context = read_todo()
    state = load_state()
    tasks = pick_tasks(today, state)
    print(f"Sentinel Daily Build — {today}")
    print(f"Generating {len(tasks)} commits...\n")

    done_state = {tuple(t) for t in state["completed"]}
    completed = []
    for i, (pillar, task) in enumerate(tasks, start=1):
        print(f"[{i}/{len(tasks)}] [{pillar.upper()}] {task[:60]}...")
        try:
            file_path, code, commit_msg = generate_module(
                client, pillar, task, todo_context, i
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue

        out = Path(file_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if (pillar, task) in done_state:
            n = state["iterations"].get(f"{pillar}::{task}", 1) + 1
            state["iterations"][f"{pillar}::{task}"] = n
            out = out.with_stem(f"{out.stem}_v{n}")
            commit_msg += f" (iter {n})"
        if out.exists():
            out = out.with_stem(f"{out.stem}_{i:02d}")
        out.write_text(code, encoding="utf-8")
        run(["git", "add", str(out)])
        run(["git", "commit", "-m", commit_msg])
        completed.append((pillar, task))
        if (pillar, task) not in done_state:
            state["completed"].append([pillar, task])
            done_state.add((pillar, task))
        print(f"  ✓  {out}")

    save_state(state)
    update_todo(completed)
    extra = 0
    if completed:
        run(["git", "add", str(STATE_PATH), "TODO.md"])
        run(["git", "commit", "-m", "chore: update TODO.md and build state"])
        extra = 1
    print(f"\nDone — {len(completed) + extra} commits pushed.")


if __name__ == "__main__":
    main()
