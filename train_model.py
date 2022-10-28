import torch.nn as nn
import torch.nn.functional as F
import cv2

# Vgg model
class cifar10_VGG(nn.Module):

    def __init__(self):
        super(cifar10_VGG, self).__init__()

        # Conv layers with batch norm

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)

        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        # fully connected layer with batch norm

        self.fc1 = nn.Linear(512 * 4 * 4, 128)
        self.bn9 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 64)
        self.bn10 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, 10)

        # Maxpool 2x2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # workflow
        # 1st & 2nd conv2d and bn
        out = F.elu(self.bn1(self.conv1(x)))
        out = F.elu(self.bn2(self.conv2(out)))
        # 1st pooling
        out = self.pool(out)

        # 3rd & 4th conv2d and bn
        out = F.elu(self.bn3(self.conv3(out)))
        out = F.elu(self.bn4(self.conv4(out)))
        # 2nd pooling
        out = self.pool(out)

        # 5th & 6th conv2d and bn
        out = F.elu(self.bn5(self.conv5(out)))
        out = F.elu(self.bn6(self.conv6(out)))
        # 3rd pooling
        out = self.pool(out)

        # 7th & 8th conv2d and bn
        out = F.elu(self.bn7(self.conv7(out)))
        out = F.elu(self.bn8(self.conv8(out)))

        # flatten
        out = out.view(-1, 512 * 4 * 4)

        # linear layer and bn
        out = F.elu(self.bn9(self.fc1(out)))
        out = F.elu(self.bn10(self.fc2(out)))
        out = self.fc3(out)

        return F.log_softmax(out, dim=1)  # log_softmax


# Resnet model

class ResidualBlock(nn.Module):
    def __init__(self, n_features):
        super(ResidualBlock, self).__init__()
        # Sequential Conv2d, BN, ReLu, Conv2d,BN, ReLu
        self.conv_block = nn.Sequential(

            nn.Conv2d(n_features, n_features, 3, 1, 1),
            nn.BatchNorm2d(n_features),
            nn.ReLU(inplace=True),

            nn.Conv2d(n_features, n_features, 3, 1, 1),
            nn.BatchNorm2d(n_features),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv_block(x) + x


def weights_init(model):
    # find the classname
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)


class cifar10_Resnet(nn.Module):
    def __init__(self, in_ch=3, out_ch=128, n_blocks=4):
        super(cifar10_Resnet, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_blocks = n_blocks

        # input size: in_ch x 32 x 32

        # Features
        n_features = 64

        # conv:
        # Conv2d layer structure
        features = [nn.Conv2d(in_ch, n_features, 4, 2, 1),
                    nn.BatchNorm2d(n_features),
                    nn.ReLU(inplace=True)]

        # Residual Block feature layer add
        for i in range(self.n_blocks):
            features += [ResidualBlock(n_features)]

        # Conv2d layer structure 2
        features += [nn.Conv2d(n_features, out_ch, 4, 2, 1),
                     nn.ReLU(inplace=True)]

        # feature extraction network
        self.features = nn.Sequential(*features)
        # state size: out_ch x 8 x 8

        # Classifier
        classifier = [nn.Linear(self.out_ch * 8 * 8, 128),  # 1st linear
                      nn.BatchNorm1d(128),
                      nn.ReLU(inplace=True),

                      nn.Linear(128, 64),  # 2nd linear
                      nn.BatchNorm1d(64),
                      nn.ReLU(inplace=True),

                      nn.Linear(64, 10)]  # 3rd linear

        # linear output feature extraction network
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.out_ch * 8 * 8)
        x = self.classifier(x)

        return x


class cifar10_Resnet_vgg(nn.Module):
    def __init__(self, in_ch=3, out_ch=128, n_blocks=2):
        super(cifar10_Resnet_vgg, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_blocks = n_blocks

        # input size: in_ch x 32 x 32

        # Features
        n_features = 64

        # conv:
        # Conv2d layer structure
        features = [nn.Conv2d(in_ch, n_features, 4, 2, 1),
                    nn.BatchNorm2d(n_features),
                    nn.ReLU(inplace=True)]

        # Residual Block feature layer add
        for i in range(self.n_blocks):
            features += [ResidualBlock(n_features)]

        # Conv2d layer structure 2
        features += [nn.Conv2d(n_features, out_ch, 4, 2, 1),
                     nn.ReLU(inplace=True)]

        # feature extraction network
        self.features = nn.Sequential(*features)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)

        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        # Maxpool 2x2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # state size: out_ch x 8 x 8 out_ch = 128
        # Classifier
        classifier = [nn.Linear(self.out_ch * 8 * 8, 128),  # 1st linear
                      nn.BatchNorm1d(128),
                      nn.ReLU(inplace=True),

                      nn.Linear(128, 64),  # 2nd linear
                      nn.BatchNorm1d(64),
                      nn.ReLU(inplace=True),

                      nn.Linear(64, 10)]  # 3rd linear

        # linear output feature extraction network
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        x = self.features(x)
        # 5th & 6th conv2d and bn
        x = F.elu(self.bn5(self.conv5(x)))
        x = F.elu(self.bn6(self.conv6(x)))
        # 3rd pooling
        x = self.pool(x)

        # 7th & 8th conv2d and bn
        x = F.elu(self.bn7(self.conv7(x)))
        x = F.elu(self.bn8(self.conv8(x)))

        x = x.view(-1, self.out_ch * 8 * 8)
        x = self.classifier(x)

        return x

# method for test set to measure F1 score and accuracy
def max_list(list_tmp):
    list_tmp = list(list_tmp)
    return list_tmp.index(max(list_tmp))


# method for RGB img to Lab img channel L: Lightness. a: Red/Green Value. b: Blue/Yellow Value
def RGB2Lab (img_path):
    img = cv2.imread(img_path)
    img_Lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    return img_Lab