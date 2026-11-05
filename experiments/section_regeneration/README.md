# 선택 구간 재생성 PoC (#13)

원곡의 `[start, end)` 구간을 다시 생성해 원곡과 합칩니다. 로직은 레포 루트의 `section_regen.py`(단위 테스트 `test_section_regen.py`)에 있고, 이 폴더의 `poc.py`는 실제 MusicGen으로 생성해 보는 실험 스크립트입니다.

## 처리 방식
1. `plan_section`: 문맥(최대 5초)과 tail(최대 1초) 길이 결정. 문맥 + 구간 + tail ≤ 30초
2. `select_melody`: 수정 대상 버전의 30초 멜로디에서 `[start - 문맥, end + tail]`만 잘라 0초 기준으로 이동 (#58 이전 곡은 허밍을 H 주기로 반복한 뒤 자름)
3. `generate_section`: 문맥 오디오에 이어서 멜로디 조건과 함께 생성(audiocraft 내부 API) → 앞쪽 문맥 제거
4. `merge_section`: 새로 생성한 부분에만 리미터(#20) → 시작 30ms, 끝 tail 구간 크로스페이드 → 원곡과 정확히 같은 길이(#21)

## 입력 (`inputs/`)
- `humix_test_{2,5}_range_legato.wav`: #14 후보 2(AMT 확장 + 음역·레가토 보정)로 생성한 30초 곡 (원본 녹음 아님)
- `humix_test_{2,5}_range_legato.json`: 위 곡을 생성할 때 쓴 30초 멜로디. 허밍 끝은 각각 12.12초, 11.69초

## 실행
```bash
python poc.py --takes 2 --sections 10-20 3-13 2-29 20-30 \
  --source inputs/humix_test_2_range_legato.wav inputs/humix_test_2_range_legato.json 12.12 \
  --source inputs/humix_test_5_range_legato.wav inputs/humix_test_5_range_legato.json 11.69
```
- 결과는 `outputs/`(gitignore): 병합한 30초 곡 `{source}_s{start}-{end}_t{take}.wav`, `results.jsonl`
- `--device`는 기본 `mps`(없으면 `cpu`). GPU에서는 `--device cuda`
- `xformers_stub.py`: xformers가 없는 환경에서 audiocraft를 불러오기 위한 로컬 전용 stub
- MPS에서는 seed를 고정해도 결과가 재현되지 않아서, 구간마다 여러 번(`--takes`) 생성해 판단

## 관찰
- 로컬 MPS 1회 테스트(humix_test_2, 10–20초): 문맥 5초 + 구간 10초 + tail 1초 생성에 412초, 병합 결과 정확히 30.0초, 생성 부분 peak 1.075 → 리미터 적용
