# 0. ResNet
ResNet-18을 이용하여 웨이퍼 결함 탐지를 해보았을 때, 정상 제품에서의 precision이 0.99로 매우 뛰어난 성능을 보여줬다. 하지만 Scratch, Loc 같은 결함 패턴들이 각각 0.67, 0.78로 아쉬운 정확도를 보여준다.

이를 분석해봤을 때, 단순 ResNet은 미세한 Scratch와 공정 노이즈를 구분하지 못하는 것으로 보였다. 이번 프로젝트에서는 이러한 패턴까지 잡을 수 있는 방향으로 프로젝트를 진행해보겠다.

*https://koreascience.kr/article/CFKO202532457661756.page* 논문의 아이디어를 이용하여 진행했음을 밝힙니다.

# 1. 데이터 로드
데이터 로드는 전과 같이 진행한다.
```
import kagglehub

# Download latest version
path = kagglehub.dataset_download("qingyi/wm811k-wafer-map")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

path = "/root/.cache/kagglehub/datasets/qingyi/wm811k-wafer-map/versions/1"
file_path = os.path.join(path, "LSWMD.pkl")

df = pd.read_pickle(file_path)

print(df.info())
```

# 2. 데이터 정제 및 라벨링

```
# 라벨 필터링 및 불량 데이터 추출 [cite: 71, 74]
df['failureNum'] = df.failureType.apply(lambda x: x[0][0] if len(x) > 0 else np.nan)
df_defect = df[(df['failureNum'].notnull()) & (df['failureNum'] != 'none')].copy()
```

# 3. 데이터 전처리
Multi-Branch 구조를 이식하고 좌표 정보를 결합하여 유사 패턴 오분류 문제를 해결해보겠다.

### 3-1. 이미지 규격화 및 노이즈 제거
웨이퍼 맵의 크기를 64x64로 통일하고, 미세한 결함이 노이즈에 묻지 않도록 정제한다.

```
import pandas as pd
import numpy as np
import cv2
from scipy import ndimage
from sklearn.model_selection import train_test_split

# 이미지 전처리 함수 (Size 64x64, Median Filter 적용) [cite: 34, 118]
def preprocess_wafer(wm, size=(64, 64)):
    rescaled = cv2.resize(wm, size, interpolation=cv2.INTER_NEAREST)
    filtered = ndimage.median_filter(rescaled, size=3)
    return filtered
```

### 3-2. 공간적 특징 추출 (Geometric Branch 준비)
바운딩 박스 중심 좌표, 크기, 웨이퍼 중심까지의 거리 등의 기하학적 정보를 추출한다. 이 정보가 ResNet이 놓치는 '위치 기반 분류'를 가능하게 한다.

```
#  기하학적 특징 추출 함수 (논문의 좌표 브랜치용 데이터) [cite: 35, 122]
def extract_geo(wm):
    y, x = np.where(wm == 2)
    if len(x) == 0: return np.zeros(6)
    min_x, max_x, min_y, max_y = np.min(x), np.max(x), np.min(y), np.max(y)
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    w, h = max_x - min_x, max_y - min_y
    img_center = wm.shape[0] / 2
    dist = np.sqrt((center_x - img_center)**2 + (center_y - img_center)**2)
    return np.array([center_x, center_y, w, h, dist, h/(w+1e-6)])

# 전략적 증강 함수 (논문의 3배수 증강 로직 반영) 
def get_augmented_data(subset_df, times=1):
    augmented_rows = []
    for _, row in subset_df.iterrows():
        img = row['waferMap']
        label = row['failureNum']
        # 원본 및 변환본 추가
        augmented_rows.append({'waferMap': img, 'failureNum': label})
        if times >= 2: augmented_rows.append({'waferMap': np.fliplr(img), 'failureNum': label})
        if times >= 3: augmented_rows.append({'waferMap': np.rot90(img), 'failureNum': label})
        if times >= 4: augmented_rows.append({'waferMap': np.flipud(img), 'failureNum': label})
    return pd.DataFrame(augmented_rows)

```

# 4. 데이터 증강
Flip(좌우/상하) 및 Rotation(90도 단위)을 통해 데이터를 3배로 늘려 불균형을 해소했다. 이는 특히 데이터가 적은 Scratch 패턴 탐지율을 높이는 데 매우 결정적이다.

```
# 클래스별 불균형 해소 실행 [cite: 78, 138]
final_list = []
for label in df_defect['failureNum'].unique():
    temp = df_defect[df_defect['failureNum'] == label]
    if len(temp) < 500: # Near-full 등 소수 클래스
        final_list.append(get_augmented_data(temp, times=4))
    elif len(temp) < 2000: # Scratch 등
        final_list.append(get_augmented_data(temp, times=2))
    else: # 데이터가 이미 많은 경우
        final_list.append(temp[['waferMap', 'failureNum']])

df_final = pd.concat(final_list).reset_index(drop=True)

# 최종 이미지 및 기하학적 특징 적용
df_final['waferMap_processed'] = df_final.waferMap.apply(preprocess_wafer)
df_final['geo_features'] = df_final.waferMap_processed.apply(extract_geo)

print(f"최종 준비된 데이터 수: {len(df_final)}")

```

### 시각화
본격적인 모델 설계에 앞서 데이터의 현 상태를 시각해보겠다.
```
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 증강 후 클래스별 데이터 분포 시각화 (Bar Chart)
plt.figure(figsize=(12, 6))
# 갯수 기준 내림차순 정렬
counts = df_final['failureNum'].value_counts().sort_values(ascending=False)
sns.barplot(x=counts.index, y=counts.values, palette='viridis')

plt.title('Distribution of Defect Patterns after Strategic Augmentation', fontsize=15)
plt.xlabel('Failure Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)

# 바 위에 숫자 표기
for i, v in enumerate(counts.values):
    plt.text(i, v + 50, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# 2. 전처리된 웨이퍼 패턴 샘플 시각화 (Sample Images)
unique_labels = df_final['failureNum'].unique()
n_labels = len(unique_labels)
rows = (n_labels + 3) // 4 # 4열 기준으로 행 수 계산

fig, axes = plt.subplots(rows, 4, figsize=(20, 5 * rows))
axes = axes.flatten()

for i, label in enumerate(unique_labels):
    # 각 클래스의 첫 번째 샘플 가져오기
    sample = df_final[df_final['failureNum'] == label].iloc[0]['waferMap_processed']
    axes[i].imshow(sample, cmap='viridis')
    axes[i].set_title(f"Pattern: {label}\n(Processed Sample)", fontsize=16)
    axes[i].axis('off')

# 남는 빈 서브플롯 숨기기
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
```

<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/a69e8031-9324-4a0f-9f68-7f54346c0dc5" />

# 5. 다중 브랜치 모델 설계
시각적 특징을 가진 ResNet50과 공간적 특징(MLP)을 융합하는 구조이다.

```
import torch
import torch.nn as nn
import torchvision.models as models

class MultiBranchWaferNet(nn.Module):
    def __init__(self, num_classes=8):
        super(MultiBranchWaferNet, self).__init__()
        
        # 브랜치 1: 이미지 처리 (ResNet50) [cite: 115, 118]
        self.resnet = models.resnet50(pretrained=True)
        # 흑백(1채널) 입력을 위해 첫 번째 컨볼루션 레이어 수정
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.visual_features = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # 브랜치 2: 좌표 처리 (MLP) [cite: 120, 124]
        # 입력: [center_x, center_y, w, h, dist, aspect_ratio]
        self.geo_branch = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # 특징 융합 및 분류 [cite: 114, 127]
        self.fusion_layer = nn.Sequential(
            nn.Linear(2048 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, img, geo):
        v_feat = self.visual_features(img)
        v_feat = torch.flatten(v_feat, 1)
        
        g_feat = self.geo_branch(geo)
        
        combined = torch.cat((v_feat, g_feat), dim=1)
        return self.fusion_layer(combined)

# 모델 인스턴스 생성
model = MultiBranchWaferNet(num_classes=8)
```

훈련, 검증, 테스트 데이터셋을 분할하여 평가의 신뢰도를 높였다.

```
from sklearn.model_selection import train_test_split

# 8:1:1 비율로 분할 (Stratify 옵션으로 클래스 비율 유지)
train_val_df, test_df = train_test_split(
    df_final, test_size=0.1, stratify=df_final['failureNum'], random_state=42
)

train_df, val_df = train_test_split(
    train_val_df, test_size=0.11, stratify=train_val_df['failureNum'], random_state=42
)

print(f"--- 데이터셋 분할 결과 ---")
print(f"Train set: {len(train_df)}개")
print(f"Validation set: {len(val_df)}개")
print(f"Test set: {len(test_df)}개")

# --- 데이터셋 분할 결과 ---
# Train set: 22892개
# Validation set: 2830개
# Test set: 2858개
```

# 6. Dataset & DataLoader 정의
```
# 클래스 순서를 수동으로 고정
target_names = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch']
label_map = {name: i for i, name in enumerate(target_names)}

import torch
from torch.utils.data import Dataset, DataLoader

class WaferDataset(Dataset):
    def __init__(self, df, label_map):
        # 데이터가 비어있지 않은지 확인
        self.images = torch.tensor(np.stack(df['waferMap_processed'].values), dtype=torch.float32).unsqueeze(1)
        # 기하학 특징 정규화 (매우 중요!)
        geo_data = np.stack(df['geo_features'].values)
        self.geos = torch.tensor(geo_data, dtype=torch.float32)
        self.labels = torch.tensor(df['failureNum'].map(label_map).values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.geos[idx], self.labels[idx]

# 데이터로더 생성
train_dataset = WaferDataset(train_df)
val_dataset = WaferDataset(val_df)
test_dataset = WaferDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"클래스 매핑: {train_dataset.label_map}")
```

### 학습 설정
논문에서는 mAP를 주요 지표로 삼았지만, 분류 모델에서는 CrossEntropyLoss를 기본으로 사용한다.

```
# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 손실 함수: 클래스 불균형이 남아있을 수 있으므로 CrossEntropy 사용
criterion = nn.CrossEntropyLoss()

# 옵티마이저: Adam (학습률은 0.001로 시작)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습률 스케줄러 (학습이 정체되면 감소시킴)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
```

### 학습
```
epochs = 10

for epoch in range(epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for imgs, geos, labels in train_loader:
        imgs, geos, labels = imgs.to(device), geos.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs, geos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    # 검증(Validation) 단계
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, geos, labels in val_loader:
            imgs, geos, labels = imgs.to(device), geos.to(device), labels.to(device)
            outputs = model(imgs, geos)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss/len(train_loader):.4f} | Acc: {100.*correct/total:.2f}% | Val Loss: {val_loss/len(val_loader):.4f}")
    scheduler.step(val_loss)
```











