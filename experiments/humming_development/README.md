# 짧은 허밍의 곡 전개 방식 비교 (#14)

허밍이 생성 길이(30초)보다 짧을 때 곡을 어떻게 이어 갈지 세 가지 방식을 같은 입력으로 비교합니다.

| 방식 | 파일명 | 설명 |
|---|---|---|
| 현재 방식 | `*_loop` | 허밍 H초의 chroma를 반복해서 30초를 채움(현재 서버 동작) |
| 후보 1: 2단계 생성 | `*_two_stage` | 0~H초는 `generate_with_chroma`, H~30초는 그 결과를 문맥으로 `generate_continuation`(멜로디 조건 없음) |
| 후보 2: 멜로디 확장 | `*_extended` | Anticipatory Music Transformer로 허밍을 30초까지 이어서 작곡한 뒤, 그 멜로디로 `generate_with_chroma`를 한 번 실행 |

## 실험 조건
- 샘플
  - `samples/vector1.json`, `samples/vector2.json`(약 30초 허밍). 앞 H초(5 / 10 / 20초)만 잘라서 짧은 허밍을 흉내 냄
  - `samples/humix_test_{1,2,4,5,6}.json`: 새로 녹음한 짧은 허밍(7.6~16.8초)을 #22 추출기로 뽑은 멜로디 벡터. 녹음 길이 그대로 사용(`--hum full`). 원본 녹음은 커밋하지 않음
- 텍스트 조건: `pop, happy`, seed 0, 생성 길이 30초
- MusicGen: `facebook/musicgen-melody`, audiocraft 1.3.0, 로컬 Apple Silicon(MPS)
- 멜로디 확장: `stanford-crfm/music-medium-800k`(360M, Apache-2.0), CPU, `top_p=0.95`
  - 피아노 한 트랙만 생성하도록 제한하고, 겹치는 음은 가장 높은 음만 남겨 단선율로 정리
- 멜로디 합성은 서버와 같은 방식(16kHz). 길이는 허밍 길이 H로 맞춤
  - `--synth pretty_midi`(기본): #23 이전 서버 방식. 음표마다 끊김. 첫 비교 결과(`vector*`, `humix_test_*_h*`)
  - `--synth continuous`: #23 이후 서버 방식(`melody_synth.py`). 이어지는 음표를 끊김 없이 합성

## 결과 파일 (`results/`)
청취 페이지(`results/*.html`, 브라우저로 열기)
- `compare.html`: vector1/2, 허밍 5/10/20초, 세 방식(#23 이전 합성)
- `recordings.html`: 새 녹음 5개, 세 방식(#23 이전 합성)
- `synth23.html`: 새 녹음 5개, 세 방식을 #23 합성 전후로 비교
- `corrected.html`: #23 합성으로 세 방식 + 후보 2 보정 버전 비교
- 녹음 페이지의 "원본 녹음" 플레이어는 레포에 파일이 없어서 재생되지 않음(`*_humming.mp3`는 gitignore)

파일
- `audio/{sample}_h{H}_{method}.mp3`: 생성 결과(128kbps mp3로 변환)
- `audio/{sample}_h{H}_melody.mp3`: MusicGen에 넣은 허밍 멜로디(H초)
- `audio/{sample}_h{H}_extended_melody.mp3`: 후보 2에서 30초까지 확장한 멜로디
- `extended/*.json`, `extended/*.mid`: 확장한 멜로디 벡터
- `generation.jsonl`: 곡별 생성 시간, peak 등
- `extension.jsonl`: 멜로디 확장 시간, 음표 수
- `audio/synth23/{sample}_{method}_after.mp3`: #23 합성으로 다시 만든 곡, `{sample}_input_{before,after}.mp3`: MusicGen에 넣은 허밍 합성 전후
- `audio/corrected/{sample}_extended_{variant}.mp3`: 후보 2 보정 버전, `{sample}_{variant}_input.mp3`: 보정한 확장 멜로디(30초), `stats.json`: 페이지용 peak·생성 시간

## 실행 방법
```bash
# 1. 후보 2 멜로디 확장 (별도 venv: anticipation이 transformers==4.29.2를 고정해서 audiocraft와 충돌)
python3.12 -m venv ../../venv-amt
../../venv-amt/bin/pip install torch transformers numpy tqdm mido pretty_midi
../../venv-amt/bin/pip install --no-deps git+https://github.com/jthickstun/anticipation.git
../../venv-amt/bin/python extend_melody.py --samples samples/vector1.json samples/vector2.json --hum 5 10 20

# 2. 곡 생성 (서버와 같은 venv)
../../venv/bin/python compare.py --samples samples/vector1.json samples/vector2.json --hum 5 10 20

# 3. 커밋용 결과 정리 (outputs/ wav -> results/ mp3)
../../venv/bin/python export_results.py

# 새 녹음: 녹음 길이 그대로, #23 합성
../../venv-amt/bin/python extend_melody.py --samples samples/humix_test_*.json --hum full
../../venv/bin/python compare.py --samples samples/humix_test_*.json --hum full --synth continuous \
    --out outputs/synth23 --extended-dir outputs/extended

# 후보 2 보정 버전 (확장 구간만 보정 -> #23 합성 -> 생성, --export로 results/audio/corrected에 mp3)
../../venv/bin/python generate_corrected.py --extended results/extended/humix_test_*_extended.json \
    --variants range range_legato range_key range_key_legato --export
```
- `correct_melody.py`: 확장 멜로디 보정(음역 / 조성 / 레가토). 허밍 구간은 바꾸지 않음
- `xformers_stub.py`: xformers가 없는 Mac에서 MusicGen을 로드하기 위한 로컬 전용 stub(서버 이미지에서는 쓰지 않음)
- `speed_check.py`: 장치별 생성 속도 측정

## 관찰 (정량)
곡 생성은 로컬 MPS 기준이라 서버 GPU보다 훨씬 느립니다. 칸 값은 `생성 시간 / peak`입니다.

| 샘플 | 허밍 | loop | two_stage | extended | 확장 음표 (확장 시간, CPU) |
|---|---|---|---|---|---|
| vector1 | 5초 | 320s / 1.56 | 192s / 1.45 | 301s / 1.02 | 65 (135s) |
| vector1 | 10초 | 196s / 0.76 | 200s / 1.43 | 222s / 1.27 | 76 (164s) |
| vector1 | 20초 | 217s / 0.83 | 254s / 1.27 | 210s / 0.40 | 30 (124s) |
| vector2 | 5초 | 231s / 1.30 | 268s / 1.59 | 209s / 0.62 | 204 (1912s) |
| vector2 | 10초 | 250s / 1.02 | 264s / 0.61 | 197s / 0.61 | 101 (149s) |
| vector2 | 20초 | 230s / 1.11 | 241s / 1.03 | 201s / 0.56 | 22 (20s) |

- 첫 생성(vector1 5초 loop)의 320s에는 MPS 워밍업이 포함됨
- two_stage는 두 번 나눠 생성해도 loop와 시간이 비슷함
- extended의 확장 시간은 곡 생성과 CPU를 같이 쓰는 상태에서 잰 값. vector2 5초의 1912s는 경합과 음표 수(204개) 때문
- 18곡 중 11곡이 peak 1.0을 넘음(#20)
- 확장한 멜로디도 입력처럼 잘게 끊김: 음표 길이 중앙값 0.03~0.08초, 소리 나는 시간이 확장 길이의 10~45%
- vector2 5초 확장은 음역이 MIDI 96까지 튐(허밍 음역 50~61). 확장할 때 음역 제한이 필요할 수 있음

## 관찰: 새 녹음 + #23 합성 (`synth23.html`)
- 허밍 입력에서 음이 새로 시작되는 횟수가 줄어듦: 39→21, 15→11, 28→18, 25→11, 10→9 (humix_test_1/2/4/5/6)
- 청취: 합성 후가 훨씬 자연스럽게 이어짐(#23, PR #24 머지)
- 남은 끊김은 추출 결과의 짧은 쉼표(`pitch: 0`) 때문
- 15곡 중 6곡이 peak 1.0 초과(#20)

## 관찰: 후보 2 확장 멜로디 보정 (`corrected.html`)
확장 구간이 허밍과 따로 노는 원인을 확장 멜로디에서 확인

| 녹음 | 허밍 음역 | 확장 음역 | 음역 밖 | 조성 밖 | 확장 구간 끊김 (초당) |
|---|---|---|---|---|---|
| humix_test_1 | 54–68 | 43–70 | 26% | 21% | 3.0 |
| humix_test_2 | 56–65 | 55–64 | 2% | 58% | 5.9 |
| humix_test_4 | 51–67 | 59–67 | 0% | 34% | 2.3 |
| humix_test_5 | 52–68 | 36–60 | 26% | 18% | 1.2 |
| humix_test_6 | 52–63 | 43–63 | 10% | 43% | 3.3 |

- 음역: 허밍보다 한 옥타브 가까이 내려가거나 올라감(humix_test_5)
- 끊김: AMT는 피아노처럼 음을 떼서 써서, 사인파로 넣으면 허밍 구간보다 2–6배 자주 끊김
- 조성: 허밍 자체도 추정 조성에서 15–22% 벗어나 있어(반음 흔들림, 음정 하강) 조 추정을 믿기 어려움

보정(`correct_melody.py`, 허밍 이후 구간만)
- 음역: 허밍 음역 ±2반음 밖의 음을 옥타브 단위로 안으로
- 레가토: 0.3초 미만 빈틈을 앞 음을 늘려 메움 → 확장 구간 끊김 초당 1.2–5.9 → 0.1–0.9
- 조성: 추정 장조 스케일 밖의 음을 반음 옮김. humix_test_2에서 0.04초 반음 트릴(D–D#)이 같은 음 연타(D#–D#)로 바뀌어 끊기고 반복되는 소리가 됨. 레가토를 같이 쓰면 연타는 긴 음 하나로 이어지지만 단조로워짐

## 청취 평가
- 1차(1인): two_stage는 허밍 이후 전개가 애매해서 loop가 훨씬 나음
- 2차(새 녹음, #23 합성): 후보 2가 괜찮다는 의견. 다만 확장 구간에서 뜬금없는 멜로디가 나온다는 의견도 있음
- 3차(보정): 음역+조성 보정은 좋게 들리는 곡도 있지만 humix_test_2처럼 끊기고 같은 음이 반복되는 경우가 있음. 위험이 적은 음역+레가토로 진행하기로 함

## 결론
- **후보 2(멜로디 확장) + 음역·레가토 보정**으로 진행. 조성 보정은 넣지 않음
- 서버 적용은 #25, 버전별 확장 멜로디 저장은 Humix-Project/humix-backend#58
- 남은 과제
  - 확장 구간 멜로디 다시 뽑기(AMT infilling)는 #13 구간 재생성과 함께
  - 출력 클리핑은 #20 리미터로 해결
  - 허밍 쪽 ±1반음 흔들림·짧은 쉼표 정리는 추출 단계에서 별도로 검토
