from torchvision.models import mobilenet_v2
from torch import nn
class MobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        # 修改首层卷积适配32x32输入
        self.model = mobilenet_v2(pretrained=False)
        self.model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        # 修改分类层
        self.model.classifier[1] = nn.Linear(1280, 10)

    def forward(self, x):
        return self.model(x)