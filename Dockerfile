# 1. Base Image: 경량화 및 추후 GPU 연산 환경 확장을 고려한 파이썬 이미지 사용
FROM python:3.11-slim

# 2. System Dependencies (추후 오디오 처리 라이브러리인 ffmpeg 등 설치 공간)
# RUN apt-get update && apt-get install -y ffmpeg

# 3. Workdir 설정
WORKDIR /app

# 4. Python 패키지 캐시 무효화 및 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 비동기 핸들러 로직 복사
COPY handler.py .

# 6. Container Entry Point
# -u 옵션을 통해 버퍼링 없이 RunPod 대시보드로 실시간 로그를 스트리밍합니다.
CMD [ "python", "-u", "/app/handler.py" ]