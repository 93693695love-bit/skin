import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import time
from PIL import ImageFile

# 이미지 손상 에러(Truncated Image) 방지
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 1. 하드웨어 가속기 설정 (GPU 우선, 없으면 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 시스템 알림: 현재 [{device}] 장치에서 연산을 수행합니다.")

# 2. 데이터 전처리 파이프라인 (Data Augmentation 포함)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), # 데이터 증강을 통한 일반화 성능 향상
        transforms.ToTensor(),
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
    
    # 전체 데이터셋 로드 (샘플링 제거)
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    
    # 병렬 처리를 위해 num_workers 설정 (서버 환경 최적화)
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    print(f"📊 학습 데이터: {dataset_sizes['train']}장 / 검증 데이터: {dataset_sizes['val']}장")

    # 3. 모델 정의 (ResNet18)
    model = models.resnet18(weights='DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    # 4. 손실 함수 및 최적화 알고리즘
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. 정식 학습 루프 (10 Epoch 이상 권장)
    num_epochs = 10
    best_acc = 0.0
    
    print(f"🔥 정식 학습 시작 (총 {num_epochs} 에포크)")
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 가장 성능이 좋은 가중치 파일 저장
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'skin_model_best.pth')

    print(f'🏁 학습 완료! 최고 검증 정확도: {best_acc:.4f}')
    # 마지막 상태도 저장
    torch.save(model.state_dict(), 'skin_model_final.pth')

if __name__ == "__main__":
    if os.path.exists('./data/train'):
        train_model()
    else:
        print("Error: './data/train' 디렉토리를 찾을 수 없습니다.")
