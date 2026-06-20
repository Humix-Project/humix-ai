from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vector_processor import MelodyProcessor

app = FastAPI(title="Humix AI Vectorization Server")
processor = MelodyProcessor()

# Spring Boot가 전달할 Request Body 규격 정의
class MelodyExtractRequest(BaseModel):
    s3_url: str

@app.post("/api/v1/ai/melody-extract")
def extract_melody(payload: MelodyExtractRequest):
    try:
        # Pipeline 1: 오디오 다운샘플링 및 로드
        signal = processor.preprocess_audio(payload.s3_url)

        # Pipeline 2: pYIN F0 주파수 궤적 추적
        f0_data = processor.extract_f0(signal)

        # Pipeline 3: 임계값 필터링 및 평균율 변환, 중간값 평활화
        n_raw = processor.hz_to_midi(f0_data)

        # Pipeline 4: 정수 양자화 및 동일 음정 병합 압축 연산
        result_vector = processor.quantize_and_map(n_raw)

        # 최종 정형화 배열 반환 (Spring Boot의 List<Map<String, Object>>로 쏙 들어갑니다)
        return result_vector

    except Exception as e:
        # 예기치 못한 에러 발생 시 상태코드 500과 사유를 로깅 반환
        raise HTTPException(status_code=500, detail=f"AI 멜로디 추출 엔진 연산 실패: {str(e)}")