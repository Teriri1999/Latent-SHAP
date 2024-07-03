import torch

# 加载.pth文件
model_path = r"F:\stylegan2-pytorch-master\lpips\weights\v0.1\vgg.pth"
model = torch.load(model_path)

# 查看模型结构
print(model)

# 查看模型参数
for name, param in model.items():
    print(f'{name}: {param}')
