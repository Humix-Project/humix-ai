import numpy as np
import librosa
import scipy.signal

class MelodyProcessor:
    def __init__(self):
        # 알고리즘 엔진 표준 샘플링 레이트: 16,000Hz
        self.target_sr = 16000
        # 0.1초 단위 추적을 위한 hop_length (16,000Hz * 0.1s = 1,600 samples)
        self.hop_length = 1600
        # 주파수 분석 창 크기 (6,400 samples)
        self.frame_length = 6400

    def preprocess_audio(self, file_path_or_url: str) -> np.ndarray:
        """
        [단계 1] 원보 오디오 전처리 (MelodyExtractor.preprocess)
        입력 환경 편차를 최소화하기 위해 16,000Hz 모노 오디오로 다운샘플링 수행
        """
        import tempfile
        import os
        import requests

        local_path = file_path_or_url
        is_url = file_path_or_url.startswith("http://") or file_path_or_url.startswith("https://")
        
        if is_url:
            print(f"Downloading audio from URL: {file_path_or_url}")
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_file.close()
            local_path = temp_file.name
            try:
                response = requests.get(file_path_or_url, stream=True)
                response.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                if os.path.exists(local_path):
                    os.remove(local_path)
                raise RuntimeError(f"Failed to download audio from URL: {e}")

        try:
            signal, _ = librosa.load(local_path, sr=self.target_sr, mono=True)
            return signal.astype(np.float32)
        finally:
            if is_url and os.path.exists(local_path):
                os.remove(local_path)

    def extract_f0(self, signal: np.ndarray) -> list[dict]:
        """
        [단계 2] pYIN 기반 주파수 추출 (MelodyExtractor.extract_f0)
        가창 음성 음고(F0) 추적 및 0.1초 단위 타임스탬프 시계열 정렬
        """
        # 가창 주파수 분석 범위 제한 (C2 ~ C7 기본값 공식 권장 반영)
        fmin = librosa.note_to_hz('C2')
        fmax = librosa.note_to_hz('C7')

        # librosa.pyin 알고리즘 구동
        f0, voiced_flag, voiced_prob = librosa.pyin(
            signal,
            fmin=fmin,
            fmax=fmax,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            sr=self.target_sr,
            fill_na=0.0 # 주파수 미검출 구간은 우선 0.0으로 채움
        )

        # 0.1초 단위의 균일한 시계열 그리드로 타임스탬프 생성
        timestamps = np.arange(len(f0)) * 0.1

        # 설계서 명세 <표 2-1> 데이터 세트 구조로 패킹
        f0_data_list = []
        for t, f, p in zip(timestamps, f0, voiced_prob):
            f0_data_list.append({
                "Timestamp": t,
                "F0_Frequency": float(f),
                "Voicing_Prob": float(p)
            })
        return f0_data_list

    def hz_to_midi(self, f0_data: list[dict]) -> np.ndarray:
        """
        [단계 3] 소음 필터링 및 표준 평균율 수식 변환 (MelodyVectorizers.hz_to_midi)
        """
        frequencies = np.array([d["F0_Frequency"] for d in f0_data], dtype=np.float32)
        probabilities = np.array([d["Voicing_Prob"] for d in f0_data], dtype=np.float32)

        # 시스템 기준 임계값(0.6) 미만이거나 주파수가 0 이하인 구간을 무음(잡음)으로 1차 처리
        invalid_mask = (probabilities < 0.6) | (frequencies <= 0) | np.isnan(frequencies)

        # log2 계산 시 0이 들어가 에러가 나는 것을 막기 위해 안전용 주파수(440Hz) 임시 대입
        safe_freq = np.where(invalid_mask, 440.0, frequencies)

        # 설계서에 명시된 표준 평균율 역산 수식 적용
        n_raw = 69.0 + 12.0 * np.log2(safe_freq / 440.0)

        # 잡음/무음 구간으로 판정되었던 마스크 자리를 완전한 0번(소리 없음)으로 치환
        n_raw[invalid_mask] = 0.0

        # [이상치 평활화 제약 조건] 커널 크기=3의 중간값 필터(Median Filter) 적용
        # 0.1초 미만의 순간적인 피치 튐 노이즈를 부드럽게 무력화합니다.
        n_raw_smoothed = scipy.signal.medfilt(n_raw, kernel_size=3)

        return n_raw_smoothed

    def quantize_and_map(self, n_raw: np.ndarray) -> list[dict]:
        """
        [단계 4] 양자화 및 음길이 계산 압축 (MelodyVectorizers.quantize_and_map)
        """
        # 실수형 MIDI 음정 배열을 반올림하여 정수형 MIDI 번호(피아노 건반 번호)로 양자화
        midi_notes = np.round(n_raw).astype(int)

        finalized_melody_vector = []
        if len(midi_notes) == 0:
            return finalized_melody_vector

        # 0.1초 그리드 연속 프레임 집계를 위한 초기화
        current_pitch = midi_notes[0]
        duration = 0.1

        # 동일한 노트가 연속되는 프레임 수를 집계하여 음길이 계산
        for i in range(1, len(midi_notes)):
            midi_num = midi_notes[i]

            if midi_num == current_pitch:
                duration += 0.1
            else:
                # 음정이 바뀌면 이전까지 쌓인 음표 정보를 리스트에 보관
                finalized_melody_vector.append({
                    "start_time_seconds": 0.0, # 자바 Spring Boot 레이어에서 정밀 정렬하므로 0.0 초기화
                    "pitch": int(current_pitch),
                    "duration_seconds": round(duration, 1) # 소수점 오차 절사
                })
                current_pitch = midi_num
                duration = 0.1

        # 루프 종료 후 남아있는 마지막 음표 처리
        finalized_melody_vector.append({
            "start_time_seconds": 0.0,
            "pitch": int(current_pitch),
            "duration_seconds": round(duration, 1)
        })

        return finalized_melody_vector