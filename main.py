import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify
from PIL import Image
import io
import os

# 1. Flask 시스템 초기화
app = Flask(__name__)

# 2. 하드웨어 가속기 설정
# 로컬 노트북에 GPU가 없어도 map_location 옵션으로 CPU 추론이 가능하게 설계했습니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  시스템 알림: 현재 추론 장치는 [{device}]입니다.")

# 3. 모델 아키텍처 정의 및 로직 객체화 (Singleton)
class SkinModel:
    def __init__(self, model_path, num_classes=2):
        # 학습 시 사용한 ResNet18 뼈대 생성
        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        # 가중치 파일 로드 및 장치 맵핑
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            self.model.to(device)
            self.model.eval() # 추론 전용 모드(Evaluation Mode) 활성화
            print(f"✅ 가중치 로드 성공: {model_path}")
        else:
            raise FileNotFoundError(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        
        self.class_names = ['acne', 'healthy']

    def predict(self, image_bytes):
        # 학습 시와 동일한 전처리 파이프라인 (224x224 리사이징 및 정규화)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 바이너리 스트림을 이미지 객체로 변환
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # 그래디언트 계산을 차단하여 메모리 효율 및 속도 향상
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, preds = torch.max(probabilities, 0)
            
        return self.class_names[preds.item()], confidence.item()

# 4. 모델 전역 로드 (서버 시작 시 1회 수행)
# 지원님이 획득한 가장 정확한 모델 파일명을 지정합니다.
MODEL_FILE = "skin_model_best.pth"
try:
    predictor = SkinModel(MODEL_FILE)
except Exception as e:
    print(f"⚠️ 시스템 경고: 모델 로드 중 오류가 발생했습니다. ({e})")
    predictor = None

# 5. API 엔드포인트 설계
@app.route("/predict", methods=["POST"])
def predict_skin():
    if not predictor:
        return jsonify({"status": "error", "detail": "모델이 준비되지 않았습니다."}), 500
        
    if 'file' not in request.files:
        return jsonify({"status": "error", "detail": "업로드된 파일이 없습니다."}), 400
        
    file = request.files['file']
    
    try:
        # 이미지 데이터 추론 수행
        img_bytes = file.read()
        label, confidence = predictor.predict(img_bytes)
        
        # 프론트엔드에 전달할 JSON 결과 구성
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
    return jsonify({"message": "SkinGuard Flask Inference Server is active."})

# 6. 서버 구동 설정
if __name__ == "__main__":
    # 로컬 개발 환경(127.0.0.1)의 8000번 포트에서 대기합니다.
    app.run(host="127.0.0.1", port=8000, debug=True)
