import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import time
from PIL import ImageFile

# 1. 시스템 무결성 설정
# 이미지 손상 에러(Truncated Image) 방지: I/O 과정에서 불완전한 파일도 로드 허용
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 하드웨어 가속기(CUDA) 컨텍스트 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 시스템 알림: 현재 [{device}] 장치에서 연산을 수행합니다.")

# 2. 데이터 전처리 파이프라인 (일반화 성능 향상을 위한 Augmentation 강화)
# 인터넷 사진과 학습 데이터 간의 도메인 격차(Domain Gap)를 줄이기 위한 전략입니다.
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),      # 리사이징 전 여유 공간 확보
        transforms.RandomResizedCrop(224), # 무작위로 확대 및 자르기 (위치 불변성 확보)
        transforms.RandomHorizontalFlip(),  # 좌우 반전
        transforms.RandomRotation(25),      # 무작위 회전 (각도 변화 대응)
        # 중요: 조명 및 필터에 민감한 인터넷 사진 대응을 위한 색상 변조
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        # ImageNet 표준 정규화 (전이 학습의 핵심 시스템 콜)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def train_model():
    data_dir = './data'
    
    # 3. 데이터 로더 구축 (FileSystem I/O 최적화)
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    
    # num_workers를 조절하여 CPU와 GPU 간의 데이터 병목(Bottleneck) 현상 방지
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    print(f"📊 학습 데이터: {dataset_sizes['train']}장 / 검증 데이터: {dataset_sizes['val']}장")

    # 4. 모델 아키텍처 정의 (ResNet18 기반 Transfer Learning)
    # weights='DEFAULT'를 사용하여 검증된 가중치로 초기화
    model = models.resnet18(weights='DEFAULT')
    num_ftrs = model.fc.in_features
    # 출력층을 우리 서비스 규격(2개 클래스: acne, healthy)으로 변경
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    # 5. 손실 함수 및 최적화 알고리즘 (Loss & Optimizer)
    criterion = nn.CrossEntropyLoss()
    # Adam 최적화 도구를 사용하되, 학습률(Learning Rate)을 미세하게 조정
    optimizer = optim.Adam(model.parameters(), lr=0.0001) 

    # 6. 학습 루프 (Training Loop)
    num_epochs = 15 # 일반화 성능을 위해 15 에포크 이상 권장
    best_acc = 0.0
    
    print(f"🔥 정밀 학습 가동 (총 {num_epochs} 에포크)")
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # 학습 모드: Dropout 및 BN 활성화
            else:
                model.eval()  # 평가 모드: 모델 상태 고정

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad() # 이전 연산의 기울기 초기화

                # Forward Pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward Pass (학습 단계에서만 수행)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 통계 계산
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 7. 체크포인트 저장 (Best Weights)
            # 검증 정확도가 갱신될 때마다 시스템 자산(pth)으로 영구 저장
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'skin_model_best.pth')
                print(f"✨ 최고 정확도 갱신: {best_acc:.4f} (가중치 저장 완료)")

    print(f'🏁 모든 프로세스 완료. 최고 검증 정확도: {best_acc:.4f}')
    torch.save(model.state_dict(), 'skin_model_final.pth')

if __name__ == "__main__":
    if os.path.exists('./data/train'):
        train_model()
    else:
        print("❌ 시스템 에러: ./data 경로가 존재하지 않습니다. 전처리를 먼저 수행하십시오.")
