import requests

# 1. 시스템 엔드포인트 설정
# localhost(127.0.0.1)의 8000번 포트로 POST 요청을 보냅니다.
url = "http://127.0.0.1:8000/predict"

# 2. 테스트용 이미지 경로 지정
# 프로젝트 폴더 내에 있는 실제 이미지 파일명을 입력하세요.
image_path = "test_image.jpg" 

def test_inference():
    try:
        # 파일 바이너리 읽기 및 스트림 생성
        with open(image_path, "rb") as f:
            # Flask 서버의 request.files['file'] 키와 일치시켜 전송
            files = {"file": f}
            
            print(f"🚀 {image_path} 데이터를 서버로 전송 중...")
            response = requests.post(url, files=files)
            
        # 3. HTTP 상태 코드 및 결과 분석
        if response.status_code == 200:
            result = response.json()
            print("\n" + "="*40)
            print("✅ AI 분석 결과 수신 성공")
            print("-" * 40)
            print(f"📍 판독 결과: {result['prediction']}")
            print(f"🎯 확신도  : {result['confidence']}%")
            print(f"💬 메시지  : {result['result_msg']}")
            print("="*40)
        else:
            print(f"❌ 서버 응답 에러: {response.status_code}")
            print(f"상세 내용: {response.json().get('detail')}")

    except FileNotFoundError:
        print(f"❌ 에러: 테스트할 이미지 파일('{image_path}')이 폴더에 없습니다.")
    except Exception as e:
        print(f"❌ 연결 실패: {e}")

if __name__ == "__main__":
    test_inference()
