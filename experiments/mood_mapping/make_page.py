"""Listening page for run_experiment.py: python make_page.py [outputs dir] -> outputs/report.html (MP3 embedded)."""
import base64
import html
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

LABELS = {'A_text': 'A. 지금 방식 (텍스트: 장르, 분위기)',
          'B_text_meta': 'B. 텍스트에 BPM·조성 추가',
          'C_melody': 'C. 멜로디 템포·조성 수정'}


def mp3_uri(wav):
    mp3 = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', str(wav), '-codec:a', 'libmp3lame', '-b:a', '128k',
                          '-f', 'mp3', '-'], capture_output=True, check=True).stdout
    return 'data:audio/mpeg;base64,' + base64.b64encode(mp3).decode()


def player(wav):
    return f'<audio controls preload="metadata" src="{mp3_uri(wav)}"></audio>'


def mean(xs):
    return sum(xs) / len(xs) if xs else float('nan')


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / 'outputs')
    rows = [json.loads(line) for line in (out / 'results.jsonl').open()]

    # Summary per condition: tempo error vs target, mode match, onset rate split by arousal
    summary = defaultdict(lambda: {'err': [], 'mode': [], 'nps_hi': [], 'nps_lo': []})
    for r in rows:
        s = summary[r['condition']]
        t = r['targets']
        s['err'].append(abs(r['output']['bpm'] - t['bpm']))
        s['mode'].append(r['output']['key']['mode'] == t['mode'])
        (s['nps_hi'] if t['arousal'] >= 0.5 else s['nps_lo']).append(r['output']['onsets_per_sec'])
    summary_html = ''.join(
        f"<tr><td>{LABELS[c]}</td><td>{mean(s['err']):.1f}</td><td>{sum(s['mode'])}/{len(s['mode'])}</td>"
        f"<td>{mean(s['nps_hi']):.2f}</td><td>{mean(s['nps_lo']):.2f}</td></tr>"
        for c, s in sorted(summary.items()))

    groups = defaultdict(dict)
    for r in rows:
        groups[(r['sample'], r['mood'])][r['condition']] = r
    sections = []
    for (sample, mood), conds in groups.items():
        t = next(iter(conds.values()))['targets']
        cells = []
        for c in LABELS:
            r = conds.get(c)
            if not r:
                cells.append('<td>-</td>')
                continue
            o = r['output']
            ok_mode = '✅' if o['key']['mode'] == t['mode'] else '❌'
            cells.append(
                f"<td><div class=t>{html.escape(r['text'])}</div>{player(out / r['file'])}"
                f"<div class=m>입력 멜로디 {r['input_bpm']} BPM · {r['input_key']}<br>"
                f"<b>결과 {o['bpm']} BPM · {o['key']['name']} {ok_mode}</b> · 초당 음 {o['onsets_per_sec']}</div>"
                f"<details><summary>입력 멜로디 듣기</summary>{player(out / r['file'].replace('.wav', '_melody.wav'))}"
                f"</details></td>")
        sections.append(
            f"<h3>{sample} · {mood} <small>(V {t['valence']}, A {t['arousal']}) → 목표 {t['bpm']} BPM, "
            f"{t['target_key']} / 허밍 {t['hum_bpm']} BPM, {t['hum_key']['name']} "
            f"(확신 차이 {t['hum_key']['margin']}), 템포 배율 ×{t['tempo_ratio']}</small></h3>"
            f"<table><tr>{''.join(f'<th>{LABELS[c]}</th>' for c in LABELS)}</tr><tr>{''.join(cells)}</tr></table>")

    page = f"""<!doctype html><html lang=ko><head><meta charset=utf-8><title>분위기별 템포·조성 실험</title>
<style>body{{font-family:-apple-system,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;width:100%;margin-bottom:8px}}td,th{{border:1px solid #ddd;padding:8px;vertical-align:top;
font-size:13px}}th{{background:#f4f4f4}}audio{{width:100%}}.t{{font-family:monospace;color:#555;margin-bottom:4px}}
.m{{margin-top:4px}}h3 small{{font-weight:normal;color:#666;font-size:13px}}</style></head><body>
<h1>분위기(V-A)별 템포·조성: 텍스트 vs 멜로디 수정</h1>
<p>같은 허밍으로 세 가지 방식을 비교. 생성 10초, CPU, seed 0. 템포·조성은 librosa로 측정(10초라 BPM은 2배/절반으로 잘못 잡힐 수 있음).</p>
<h2>요약</h2><table><tr><th>방식</th><th>목표 BPM과의 평균 차이</th><th>장/단조 일치</th>
<th>초당 음 (각성도 높은 분위기)</th><th>초당 음 (각성도 낮은 분위기)</th></tr>{summary_html}</table>
<h2>곡별 비교</h2>{''.join(sections)}</body></html>"""
    (out / 'report.html').write_text(page)
    print(out / 'report.html')


if __name__ == '__main__':
    main()
