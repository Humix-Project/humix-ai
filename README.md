# 🤖 Humix - AI Processing Worker

Humix 프로젝트의 딥러닝 모델 서빙과 디지털 신호 처리(DSP)를 전담하는 비동기 서버리스 AI 연산 워커입니다. 사용자의 음성 허밍을 분석하여 멜로디 벡터를 추출하고, 조건부 생성형 AI 모델을 통해 최종 음악을 렌더링합니다.

## 🛠 Tech Stack
* **Language/Framework:** Python 3.10, FastAPI
* **AI/ML:** PyTorch, Meta MusicGen, Pop Music Transformer, Yin-Yang Framework
* **DSP:** Librosa (pYIN Algorithm)
* **Infrastructure:** RunPod Serverless (Cloud GPU Worker), Docker

## 🚀 Core Pipelines & Algorithms

### 1. 비동기 추론 엔진 (Serverless Worker)
* **Event-Driven Architecture:** 메인 서버의 Job Queue에서 요청을 수신할 때만 GPU 자원을 할당받아 연산 비용을 최적화합니다.
* 외부 LLM API와 연동하여 사용자의 텍스트 입력을 MusicGen 모델이 이해하기 쉬운 형태의 프롬프트로 가공합니다.

### 2. 멜로디 추출 및 벡터화 모듈 (DSP)
사용자의 비정형 오디오 신호에서 유효한 피치를 추출하여 표준 규격의 데이터로 변환합니다.
* **전처리:** 모든 입력 오디오를 16,000Hz 모노 오디오로 다운샘플링하여 품질 편차를 최소화합니다.
* **pYIN 알고리즘 (`extract_f0`):** 허밍의 기본 주파수(F0)를 0.1초 단위로 정밀하게 추적합니다.
* **벡터화 및 양자화 (`hz_to_midi`, `quantize_and_map`):** 소음 구간을 필터링하고 연속된 주파수(Hz)를 표준 평균율 수식을 적용해 정수형 MIDI 노트(`C4`, `E4` 등) 및 Duration으로 변환하여 Phrase JSON 객체로 정형화합니다.

### 3. 생성형 AI 기반 곡 생성 파이프라인
추출된 멜로디 벡터를 기반으로 MusicGen 및 Yin-Yang 프레임워크를 활용해 편곡을 수행합니다.
* **Yin-Yang Framework (선율 확장 & 변주):** * `PhraseCodec` 및 `YinYangCodec`을 통해 객체를 REMI 토큰 시퀀스로 양방향 변환합니다.
    * **Pop Music Transformer**를 활용해 단일 Phrase 단위의 선율 확장을 수행합니다.
    * **Genre Refiner**를 거쳐 확장된 선율이 목표 장르의 음악적 문법을 따르도록 스타일을 교정합니다.
* **Meta MusicGen (오디오 렌더링):** * 멜로디 텐서 데이터와 LLM으로 최적화된 프롬프트(`modern pop`, `clean production` 등)를 결합하여 조건부 생성을 수행합니다.
    * 사용자가 지정한 특정 구간만 재생성하는 인페인팅(Inpainting) 기능을 지원합니다.