# 짧은 허밍의 곡 전개 방식 비교 (#14)

허밍이 생성 길이(30초)보다 짧을 때 곡을 어떻게 이어 갈지 세 가지 방식을 같은 입력으로 비교합니다.

| 방식 | 파일명 | 설명 |
|---|---|---|
| 현재 방식 | `*_loop` | 허밍 H초의 chroma를 반복해서 30초를 채움(현재 서버 동작) |
| 후보 1: 2단계 생성 | `*_two_stage` | 0~H초는 `generate_with_chroma`, H~30초는 그 결과를 문맥으로 `generate_continuation`(멜로디 조건 없음) |
| 후보 2: 멜로디 확장 | `*_extended` | Anticipatory Music Transformer로 허밍을 30초까지 이어서 작곡한 뒤, 그 멜로디로 `generate_with_chroma`를 한 번 실행 |

## 실험 조건
- 샘플: `samples/vector1.json`, `samples/vector2.json`(약 30초 허밍). 앞 H초(5 / 10 / 20초)만 잘라서 짧은 허밍을 흉내 냄
- 텍스트 조건: `pop, happy`, seed 0, 생성 길이 30초
- MusicGen: `facebook/musicgen-melody`, audiocraft 1.3.0, 로컬 Apple Silicon(MPS)
- 멜로디 확장: `stanford-crfm/music-medium-800k`(360M, Apache-2.0), CPU, `top_p=0.95`
  - 피아노 한 트랙만 생성하도록 제한하고, 겹치는 음은 가장 높은 음만 남겨 단선율로 정리
- 멜로디 합성은 서버(`handler.convert_vectors_to_wav_tensor`)와 같은 방식(사인파, 16kHz). 길이는 허밍 길이 H로 맞춤

## 결과 파일 (`results/`)
- `audio/{sample}_h{H}_{method}.mp3`: 생성 결과(128kbps mp3로 변환)
- `audio/{sample}_h{H}_melody.mp3`: MusicGen에 넣은 허밍 멜로디(H초)
- `audio/{sample}_h{H}_extended_melody.mp3`: 후보 2에서 30초까지 확장한 멜로디
- `extended/*.json`, `extended/*.mid`: 확장한 멜로디 벡터
- `generation.jsonl`: 곡별 생성 시간, peak 등
- `extension.jsonl`: 멜로디 확장 시간, 음표 수

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
```
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

## 청취 평가
- 1차(1인): two_stage는 허밍 이후 전개가 애매해서 loop가 훨씬 나음
- [ ] 허밍 구간 이후에도 주제가 유지되는지
- [ ] 허밍이 끝나는 지점의 전환이 자연스러운지
- [ ] 음악적 자연스러움(블라인드 청취, 여러 명)

## 결론
- 미정(청취 평가 후 작성)
