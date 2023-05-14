import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
        # 첫 번째 컨볼루션 레이어
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # 두 번째 컨볼루션 레이어
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # 활성화 함수
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 첫 번째 컨볼루션 레이어의 출력 feature map
        x = self.conv1(x)
        conv1_out = self.relu(x)
        
        # 두 번째 컨볼루션 레이어의 출력 feature map
        x = self.conv2(conv1_out)
        conv2_out = self.relu(x)
        
        return conv1_out, conv2_out

# 입력 이미지 생성
inputs = torch.randn((1, 3, 82, 82))

# 모델 생성
model = MyModel()

# 모델 적용
conv1_out, conv2_out = model(inputs)

# 첫 번째 feature map 시각화
fig, ax = plt.subplots()
ax.imshow(conv1_out[0, 0].detach().numpy())
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Conv1 output')
plt.savefig('conv1_out.png', bbox_inches='tight')

# 두 번째 feature map 시각화
fig, ax = plt.subplots()
ax.imshow(conv2_out[0, 0].detach().numpy())
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Conv2 output')
plt.savefig('conv2_out.png', bbox_inches='tight')
