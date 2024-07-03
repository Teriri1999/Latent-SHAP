# import torch
# import torchvision.models as models
#
#
# vgg = models.vgg16(pretrained=False, num_classes=10)
#
# model_path = r"F:\stylegan2-pytorch-master\lpips\weights\v0.1\vgg16.pth"
# vgg.load_state_dict(torch.load(model_path))
#
# weights = []
# for layer in vgg.features[:5]:
#     if isinstance(layer, torch.nn.Conv2d):
#         weights.append(layer.weight)
#
# # 将提取到的权重保存到文件中
#
# if len(weights) < 3:
#     print('Error: The weights list does not have at least 3 elements.')
# else:
#     torch.save({'lin0.model.1.weight': weights[0],
#                 'lin1.model.1.weight': weights[1],
#                 'lin2.model.1.weight': weights[2],
#                 'lin3.model.1.weight': weights[3],
#                 'lin4.model.1.weight': weights[4]}, 'vgg_SAR.pth')


import torch
import torchvision.models as models


vgg = models.vgg16(pretrained=True)
my_vgg = models.vgg16(pretrained=False)
my_vgg.features = vgg.features

weights = []
for layer_idx in [0, 3, 6, 8, 11]:
    layer = my_vgg.features[layer_idx]
    if isinstance(layer, torch.nn.Conv2d):
        weights.append(layer.weight)

torch.save({'lin0.model.1.weight': weights[0],
            'lin1.model.1.weight': weights[1],
            'lin2.model.1.weight': weights[2],
            'lin3.model.1.weight': weights[3],
            'lin4.model.1.weight': weights[4]}, 'vgg_SAR.pth')
