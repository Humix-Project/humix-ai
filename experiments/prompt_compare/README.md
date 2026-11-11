# MusicGen 프롬프트 방식 비교 (#15)

같은 멜로디·시드로 genre/mood/prompt 조합마다 프롬프트 방식(`prompt_builder.STRATEGIES`)별 음악을 만들어 비교합니다.

| 방식 | 설명 |
|---|---|
| `simple` | 현재 서버 방식. `"{genre}, {mood}, {prompt}"` 단순 조합 |
| `translate` | A. Gemini로 한국어 prompt만 영어로 번역한 뒤 단순 조합 |
| `rewrite` | D. genre/mood는 템플릿, prompt는 Gemini가 MusicGen 캡션 스타일로 재작성 |

## 파일

| 파일 | 내용 |
|---|---|
| `inputs.json` | 고정 멜로디(10초)와 비교 조합 11개, 생성 길이, 시드 |
| `run_compare.py` | 조합 × 방식별 생성. 결과는 `outputs/<실행시간>/` (git 제외) |
| `make_report.py` | 결과 폴더로 `report.html`(텍스트 조건 + 오디오 내장) 생성 |
| `llm_cache.json` | Gemini 답 캐시. 같은 입력이면 API를 다시 부르지 않음 |
| `results/` | 공유용으로 커밋한 실행 결과 (MP3 내장 report) |

## 실행

저장소 루트에서 실행합니다.

```bash
# 텍스트 조건만 확인 (GPU 불필요)
python experiments/prompt_compare/run_compare.py --dry-run

# GPU(RunPod)
python experiments/prompt_compare/run_compare.py --duration 10

# Mac (Apple Silicon)
python experiments/prompt_compare/run_compare.py --device mps --duration 5 --cases c03_acoustic_ko c11_pop_sad_slang

# 커밋/공유용 MP3 report
python experiments/prompt_compare/make_report.py experiments/prompt_compare/outputs/<실행시간> --mp3 --out experiments/prompt_compare/results/<이름>/report.html
```

- `llm_cache.json`에 있는 입력은 Gemini 키 없이 동작합니다. `inputs.json`의 prompt나 `prompt_builder.py`의 지시문·예시를 바꾸면 새 LLM 호출이 필요하므로 저장소 루트 `.env`에 `GEMINI_API_KEY`를 넣으세요.
- Gemini 무료 등급은 모델당 분당 15회 제한입니다. 새 조합이 많으면 429가 나며, 재시도 후에도 실패하면 해당 행은 `simple`로 대체됩니다(로그에 `Falling back to simple`).
- Mac(16GB)에서는 MPS 메모리가 16GB 가까이 올라가 스왑을 씁니다. 5초 생성이 곡당 2~3분, 10초는 메모리 부족으로 진행되지 않았습니다. 다른 앱을 끄고 5초로 쓰거나 GPU에서 실행하세요.
