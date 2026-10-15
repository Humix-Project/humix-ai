"""Build a self-contained listening page (report.html) for one comparison run.

    python experiments/prompt_compare/make_report.py experiments/prompt_compare/outputs/<run_id>

Audio is embedded as base64, so the single HTML file can be opened or shared anywhere.
"""
import base64
import csv
import html
import json
import os
import re
import sys

STRATEGY_LABELS = {
    "simple": "지금 방식 (단순 조합)",
    "translate": "A. 번역만",
    "rewrite": "D. 템플릿 + LLM 재작성",
}

# Runs of non-ASCII text; T5 turns these into <unk>, so MusicGen never sees them
NON_ASCII_RUN = re.compile(r"[^\x00-\x7f]+(?:\s+[^\x00-\x7f]+)*")

STYLE = """
:root { --bg:#f7f7f8; --card:#fff; --text:#1d1d1f; --muted:#6e6e73; --border:#e3e3e8; --accent:#3a5bd9; --unk-bg:#fde2e1; --unk-text:#b42318; }
@media (prefers-color-scheme: dark) { :root { --bg:#141416; --card:#1f1f23; --text:#ececf1; --muted:#9a9aa3; --border:#33333a; --accent:#8ea2ff; --unk-bg:#4a1d1b; --unk-text:#ffb4ab; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.6 -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", sans-serif; }
main { max-width:960px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:22px; margin:0 0 4px; }
.meta { color:var(--muted); font-size:13px; margin-bottom:16px; }
.legend { font-size:13px; color:var(--muted); margin-bottom:24px; }
.case { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:20px; margin-bottom:20px; }
.case h2 { font-size:16px; margin:0 0 8px; }
.input { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px; }
.chip { border:1px solid var(--border); border-radius:999px; padding:2px 10px; font-size:13px; }
.chip b { color:var(--muted); font-weight:500; margin-right:4px; }
.melody { font-size:13px; color:var(--muted); margin-bottom:12px; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.row { border-top:1px solid var(--border); padding:14px 0 4px; }
.row h3 { font-size:14px; margin:0 0 6px; color:var(--accent); }
.text { font-family:ui-monospace, Menlo, monospace; font-size:13px; background:var(--bg); border-radius:8px; padding:8px 10px; margin-bottom:8px; word-break:break-word; }
mark.unk { background:var(--unk-bg); color:var(--unk-text); border-radius:4px; padding:0 2px; }
audio { width:100%; max-width:480px; height:36px; }
.time { font-size:12px; color:var(--muted); }
"""


def audio_tag(path):
    if not os.path.exists(path):
        return '<span class="time">(오디오 없음)</span>'
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f'<audio controls preload="metadata" src="data:audio/wav;base64,{encoded}"></audio>'


def highlight_unknown(text):
    escaped = html.escape(text)
    return NON_ASCII_RUN.sub(lambda m: f'<mark class="unk" title="MusicGen이 읽지 못함">{m.group(0)}</mark>', escaped)


def build_report(run_dir):
    with open(os.path.join(run_dir, "results.csv"), encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        config = json.load(f)

    cases = {}
    for row in rows:
        cases.setdefault(row["case_id"], []).append(row)

    melody = audio_tag(os.path.join(run_dir, "melody.wav"))
    sections = []
    for case_id, case_rows in cases.items():
        first = case_rows[0]
        chips = "".join(
            f'<span class="chip"><b>{label}</b>{html.escape(first[key] or "-")}</span>'
            for label, key in (("genre", "genre"), ("mood", "mood"), ("prompt", "prompt"))
        )
        strategy_rows = []
        for row in case_rows:
            seconds = f'{float(row["generate_ms"]) / 1000:.0f}초' if row["generate_ms"] else ""
            strategy_rows.append(f"""
      <div class="row">
        <h3>{html.escape(STRATEGY_LABELS.get(row["strategy"], row["strategy"]))}</h3>
        <div class="text">{highlight_unknown(row["description"])}</div>
        {audio_tag(os.path.join(run_dir, row["wav"])) if row["wav"] else ""}
        <div class="time">생성 {seconds}</div>
      </div>""")
        sections.append(f"""
    <section class="case">
      <h2>{html.escape(case_id)}</h2>
      <div class="input">{chips}</div>
      <div class="melody">입력 멜로디 {melody}</div>
      {"".join(strategy_rows)}
    </section>""")

    duration = config["inputs"].get("duration_seconds")
    page = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>프롬프트 방식 비교</title>
<style>{STYLE}</style>
</head>
<body>
<main>
  <h1>MusicGen 프롬프트 방식 비교</h1>
  <div class="meta">{html.escape(os.path.basename(os.path.normpath(run_dir)))} · {duration}초 · seed {config["inputs"].get("seed")} · 같은 조합은 같은 멜로디·시드로 생성</div>
  <div class="legend"><mark class="unk">빨간 표시</mark>는 MusicGen 텍스트 인코더(T5)가 읽지 못하고 버리는 부분이에요.</div>
  {"".join(sections)}
</main>
</body>
</html>
"""
    out_path = os.path.join(run_dir, "report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: make_report.py <run_dir>")
    print(build_report(sys.argv[1]))
