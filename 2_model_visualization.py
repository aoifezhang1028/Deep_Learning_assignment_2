import torch
from torch.autograd import Variable
from torchviz import make_dot
import train_model

# model_Yu = train_model.cifar10_VGG()
# model_Yu = train_model.cifar10_Resnet()
model_Yu = train_model.cifar10_Resnet_vgg()
x = Variable(torch.randn(128, 3, 32, 32))
y = model_Yu(x)


vis_graph = make_dot(y, params=dict(model_Yu.named_parameters()))
vis_graph.view()
