# 파이썬 3.11 가벼운 버전 사용
FROM python:3.11-slim

# 작업할 폴더 지정
WORKDIR /app

# 라이브러리 목록 복사 후 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# handler.py 파일 복사
COPY handler.py .

# 컨테이너가 켜지면 파이썬 파일 실행 (-u 옵션은 로그를 즉시 보기 위함)
CMD [ "python", "-u", "/app/handler.py" ]