# 0. ResNet
ResNet-18을 이용하여 웨이퍼 결함 탐지를 해보았을 때, 정상 제품에서의 precision이 0.99로 매우 뛰어난 성능을 보여줬다. 하지만 Scratch, Loc 같은 결함 패턴들이 각각 0.67, 0.78로 아쉬운 정확도를 보여준다.

이를 분석해봤을 때, 단순 ResNet은 미세한 Scratch와 공정 노이즈를 구분하지 못하는 것으로 보였다. 이번 프로젝트에서는 이러한 패턴까지 잡을 수 있는 방향으로 프로젝트를 진행해보겠다.

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

