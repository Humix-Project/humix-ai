import runpod
import time
import requests

def generate_music_dummy(job):
    # 1. 스프링 서버가 보낸 데이터(요청 바디) 꺼내기
    job_input = job['input']
    task_id = job_input.get('task_id', 'unknown_task')
    callback_url = job_input.get('callback_url')
    
    print(f"[로깅] 작업 시작! Task ID: {task_id}")

    # 2. 진짜 AI가 음악을 만드는 척 시간 끌기 (5초 대기)
    print("[로깅] AI가 음악을 생성하는 중입니다...")
    time.sleep(5)

    # 3. 팀원이 명세서에 적어둔 '성공' 응답 포맷 만들기
    result_data = {
        "task_id": task_id,
        "status": "SUCCESS",
        "result": {
            "generated_audio_url": "https://dummy-bucket.s3.amazonaws.com/generated/songs/fake_output.mp3",
            "duration_seconds": 30
        }
    }

    # 4. 스프링 백엔드로 결과 쏴주기 (Webhook Callback)
    if callback_url:
        try:
            response = requests.post(callback_url, json=result_data)
            print(f"[로깅] 콜백 전송 성공! (상태 코드: {response.status_code})")
        except Exception as e:
            print(f"[로깅] 콜백 전송 실패: {e}")

    # 5. RunPod 자체 시스템에 작업이 끝났음을 알림
    return {"message": "Job completed successfully"}

# RunPod 서버리스 시작 (이 한 줄이 FastAPI 대문 역할을 합니다)
runpod.serverless.start({"handler": generate_music_dummy})