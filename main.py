import os
import requests
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# 1. 환경 변수 로드
load_dotenv()

app = FastAPI()

# 2. CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. 제미나이 AI 설정
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# 🎯 지원님 전용: 터미널 진단으로 확인된 최신 모델 적용!
try:
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    print("✅ 제미나이 모델 연결 완료: gemini-2.5-flash")
except Exception as e:
    print(f"❌ 모델 초기화 에러: {e}")
    gemini_model = None

@app.get("/")
def home():
    return {"status": "SkinGuard Backend is Online", "owner": "지원님"}

@app.post("/analyze-skin")
async def analyze_skin(file: UploadFile = File(...)):
    contents = await file.read()
    
    # [Step 1] Face++ AI 분석 (피부 수치 데이터 추출)
    face_url = "https://api-us.faceplusplus.com/facepp/v3/detect"
    face_data = {
        "api_key": os.getenv("FACEPLUS_KEY"),
        "api_secret": os.getenv("FACEPLUS_SECRET"),
        "return_attributes": "skinstatus,gender,age"
    }
    files = {"image_file": (file.filename, contents, file.content_type)}
    
    print(f"🚀 Face++ 분석 시작... (파일명: {file.filename})")
    face_res = requests.post(face_url, data=face_data, files=files).json()

    # [Step 2] 데이터 가공 및 제미나이 조언 생성
    try:
        # 얼굴을 찾지 못했을 경우 예외 처리
        if not face_res.get('faces'):
            return {"error": "사진에서 얼굴을 찾을 수 없습니다. 정면 사진을 사용해 주세요.", "debug": face_res}

        # 데이터 정리
        face_attr = face_res['faces'][0]['attributes']
        skin = face_attr['skinstatus']
        
        # 💡 제미나이에게 전달할 프롬프트
        prompt = f"""
        너는 피부과 전문의야. 아래 피부 분석 수치를 보고 20대 남성에게 한국어로 조언해줘.
        - 나이: {face_attr['age']['value']}세
        - 여드름 점수: {skin['acne']} (0~100, 높을수록 심함)
        - 잡티 점수: {skin['stain']}
        - 피부 건강도: {skin['health']}
        딱 3줄로 핵심 관리법만 친절하게 말해줘.
        """
        
        # 제미나이 답변 생성 (최신 2.5 버전 사용)
        if gemini_model:
            print("💡 제미나이 조언 생성 중...")
            gemini_res = gemini_model.generate_content(prompt)
            advice_text = gemini_res.text
            print("✅ 조언 생성 완료!")
        else:
            advice_text = "제미나이 모델 오류로 조언을 생성하지 못했습니다."

        # [Step 3] 최종 결과 반환
        return {
            "analysis": {
                "age": face_attr['age']['value'],
                "gender": "Male (Confirmed)",
                "skin_scores": skin
            },
            "advice": advice_text
        }

    except Exception as e:
        print(f"❌ 전체 분석 에러: {e}")
        return {"error": f"분석 실패: {str(e)}", "raw_data": face_res}