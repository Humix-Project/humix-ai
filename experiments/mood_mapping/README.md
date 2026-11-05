# 분위기(V-A)별 템포·조성 반영 방식 비교 (#27)

기획의 분위기 키워드(V-A 공간 20개)에 맞춰 곡의 템포·조성을 바꾸려면 MusicGen 텍스트에 넣는 게 나은지, 허밍 음표 목록을 먼저 수정하는 게 나은지 같은 허밍으로 비교합니다. 텍스트 프롬프트 개선(#15)과 짝을 이루는 실험입니다.

| 조건 | 멜로디 | 텍스트 |
|---|---|---|
| `A_text` | 원본 | `kpop, {mood}` (현재 서버) |
| `B_text_meta` | 원본 | `kpop, {mood}. bpm: X. key: T mode` (MusicGen 학습 메타데이터 형식) |
| `C_melody` | 템포·조성 수정 | `kpop, {mood}` |

## 파일
- `mood_mapping.py`: 백엔드로 옮길 로직. 음표 목록(`pitch`, `onset_seconds`, `duration_seconds`, `pitch 0` = 쉼표)만 다룸
  - `MOODS`: 기획 문서의 V-A 좌표
  - `mood_targets`: 목표 BPM(`60 + 90 × arousal`)과 장/단조(`valence < 0.5`면 단조). 임시 값, 문헌·실험으로 보정 예정
  - `melody_key`: Krumhansl-Schmuckler 조 추정. `margin`(1·2위 상관 차이)이 작으면 확신이 낮음
  - `change_tempo`, `change_mode`: 템포 배율, 같은 으뜸음의 장조↔자연단음계(3·6·7음 반음 이동)
  - `apply_mood`: 템포 배율을 0.75~1.33으로 제한해서 허밍 원형을 지킴
- `run_experiment.py`: 허밍 × 분위기 × 조건마다 생성하고 librosa로 템포·초당 음 개수·조성을 측정해 `outputs/results.jsonl`에 기록. 끊겨도 다시 실행하면 이어서 진행
- `make_page.py`: `outputs/report.html`(MP3 내장 청취 페이지)
- `samples/`: #22 추출기로 뽑은 허밍 멜로디 벡터(녹음 원본 아님)
- `xformers_stub.py`: xformers가 없는 환경(Mac)에서 MusicGen을 불러오기 위한 stub. xformers가 있으면 아무것도 안 함

## 실행
```bash
python run_experiment.py --device cuda   # GPU
python run_experiment.py                 # CPU (기본값)
python make_page.py
```
- 기본: 허밍 `humix_test_2/4/5` × 분위기 `exciting/peaceful/sad/epic` × 3조건 = 36곡, 10초, seed 0
- 레포 루트의 `melody_synth.py`(서버와 같은 합성)를 씀
- 로컬 16GB Mac: MPS는 10초 생성에서 메모리가 18GB까지 차서 사실상 멈춤. CPU는 곡당 약 3분 반
- `outputs/`는 커밋하지 않음

## 결과 (`results/20261105_cpu_10s/`, 로컬 CPU, 36곡)
- `report.html`: 청취 페이지(브라우저로 열기, GitHub에서는 "Download raw file"), `results.jsonl`: 곡별 측정값

| 조건 | 장/단조 일치 (12곡) | 허밍과 목표 장/단조가 다른 6곡 중 일치 | 초당 음: exciting / epic / peaceful / sad |
|---|---|---|---|
| A 지금 방식 | 7 | 4 | 4.7 / 4.5 / 4.9 / **2.9** |
| B 텍스트에 BPM·조성 | 5 | 2 | 3.5 / 4.8 / 4.6 / 4.0 |
| C 멜로디 수정 | 6 | 4 | 4.1 / 3.9 / 4.8 / 4.0 |

- B(텍스트 BPM·조성)는 A보다 나아지지 않음. 장/단조 일치가 줄고, 분위기별 초당 음 차이도 사라짐
- C(멜로디 템포 ×0.75~1.33, 장/단조 변환)도 수치상 A보다 나아지지 않음. 허밍을 빠르게 한 exciting에서 초당 음이 오히려 줄었고(4.7 → 4.1), 느리게 한 peaceful은 그대로(4.9 → 4.8). 생성곡의 리듬은 허밍 속도보다 MusicGen이 깐 반주·드럼이 정하는 것으로 보임
- 분위기 차이가 가장 잘 드러난 건 A의 sad(초당 음 2.9)뿐. 지금은 텍스트의 분위기 단어가 그나마 가장 효과가 있음

측정의 한계(결론을 내기엔 노이즈가 큼)
- BPM: 10초 곡이라 beat tracker 값이 187.5 / 96.2 / 93.8 / 46.9처럼 몇 값에 몰리고 2배·절반으로 튐. 표에서 제외하고 초당 음 개수로 봄
- 조성: 허밍 조 추정 확신도가 낮음(1·2위 상관 차이 0.14~0.24). 짧은 멜로디에서는 나란한조(B major ↔ G# minor 등)를 잘 구분하지 못함. 생성곡 조성도 반주가 섞여 확신도가 0.01~0.25로 낮음
- 조건마다 1곡(seed 0)뿐이라 생성마다의 무작위 차이가 섞여 있음
- 다음에 볼 것: 청취 평가, 생성 길이 30초, seed 여러 개, 허밍 템포 추정을 BPM 대신 초당 음 개수로 바꾸기

## 참고 문헌
- Eerola, Friberg & Bresin (2013), Frontiers in Psychology 4:487. 조성 > 템포 순으로 영향이 크고, 선형 모델로 평가의 77~89%를 설명, 요소 간 상호작용 거의 없음. 템포를 초당 음 개수(NPS)로 다룸(1.2 / 2 / 2.8 / 4.4 / 6)
- Bresin & Friberg (2011), Cortex 47. 감정별 대표 템포(초당 음): 행복 5.0, 공포 4.4, 중립 3.2, 평화 2.2, 슬픔 1.3 (재구현 코드 dmwhyatt/emotion-sliders 기준, 원문 확인 필요)
- Gagnon & Peretz (2003): 단순 멜로디에서는 템포가 조성보다 강함
- Drake & Botte (1993): 템포 차이 식별 한계 약 3~6%
- Livingstone et al. (2010), CMERS: 규칙으로 악보의 조성·템포 등을 바꿔 감정을 옮김
- EMOPIA (ISMIR 2021): V-A 4사분면 팝 피아노 데이터셋, `key_mode_tempo.csv`로 매핑 보정에 활용 가능
