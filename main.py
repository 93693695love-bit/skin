import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import os

# 1. Flask 시스템 초기화 및 CORS 설정
app = Flask(__name__)
CORS(app) # 모든 도메인에서의 API 요청을 허용합니다.

# 2. 연산 장치 설정 (로컬은 보통 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  시스템 알림: 현재 추론 장치는 [{device}]입니다.")

# 3. AI 모델 아키텍처 및 추론 클래스
class SkinModel:
    def __init__(self, model_path, num_classes=2):
        # 뼈대 구축 (ResNet18)
        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        # 가중치(Weights) 로드 및 장치 맵핑
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            self.model.to(device)
            self.model.eval() # 추론 전용 모드 활성화
            print(f"✅ 가중치 로드 성공: {model_path}")
        else:
            raise FileNotFoundError(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        
        self.class_names = ['acne', 'healthy']

    def predict(self, image_bytes):
        # 학습 시와 동일한 전처리 파이프라인
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            # Softmax를 통해 각 클래스별 확률값(0~1) 산출
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, preds = torch.max(probabilities, 0)
            
        return self.class_names[preds.item()], confidence.item()

# 4. 모델 전역 객체 생성
# 서버 가동 시 딱 한 번만 메모리에 올립니다.
MODEL_FILE = "skin_model_best.pth"
try:
    predictor = SkinModel(MODEL_FILE)
except Exception as e:
    print(f"⚠️ 시스템 경고: 모델 로드 실패. ({e})")
    predictor = None

# 5. API 엔드포인트 설계 (POST /predict)
@app.route("/predict", methods=["POST"])
def predict_skin():
    if not predictor:
        return jsonify({"status": "error", "detail": "AI 모델이 로드되지 않았습니다."}), 500
        
    if 'file' not in request.files:
        return jsonify({"status": "error", "detail": "전송된 파일이 없습니다."}), 400
        
    file = request.files['file']
    
    try:
        img_bytes = file.read()
        label, confidence = predictor.predict(img_bytes)
        
        # 🛡️ 시스템 가드: 신뢰도 임계값(Confidence Threshold)
        # 85% 미만의 확신을 가진 결과는 '분석 불가'로 처리하여 고양이 등의 오답을 방어합니다.
        THRESHOLD = 0.85 
        
        if confidence < THRESHOLD:
            return jsonify({
                "status": "uncertain",
                "prediction": "unknown",
                "confidence": round(confidence * 100, 2),
                "result_msg": "피부 상태를 명확히 판별할 수 없습니다. 사진을 다시 찍어주세요."
            })
        
        # 성공 시 결과 반환
        return jsonify({
            "status": "success",
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "result_msg": f"분석 결과 {label} 상태일 확률이 {round(confidence * 100, 2)}%입니다."
        })
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"message": "SkinGuard API is active."})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
