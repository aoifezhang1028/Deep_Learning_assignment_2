from sklearn.metrics import classification_report
import torch
import torchvision.transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import train_model

model_dir = 'classification_rgb/run_6/0.788_resnet_best.pth'
model = torch.load(model_dir)
label_list = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

colour_space = 'rgb'  # rgb

# Dataset
class img_dataset(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, imag_index):
        img_name = self.img_path[imag_index]
        img_path = os.path.join(self.path, img_name)
        if colour_space == 'Lab':
            img = train_model.RGB2Lab(img_path)
        else:
            img = cv2.imread(img_path)
        img_label = self.label_dir
        img_transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize
                                                        ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        img_label = int(label_list.index(img_label))

        # img to tensor
        img = img_transform(img)
        # label to tensor
        img_label = torch.as_tensor(img_label, dtype=torch.long)

        return img, img_label

    def __len__(self):
        return len(self.img_path)


# test dataset
root_dir_test = 'classification_img_CIFAR/test'

airplane_dataset_test = img_dataset(root_dir_test, 'airplane')
full_dataset_test = airplane_dataset_test

for label in label_list[1:]:
    full_dataset_test = full_dataset_test + img_dataset(root_dir_test, str(label))

test_dataloader = DataLoader(full_dataset_test, batch_size=8, shuffle=True)

total_label = []
total_outputs = []
for data in test_dataloader:
    img, label = data
    img = img.to(device)
    label = label.to(device)
    outputs = model(img)
    outputs = outputs.tolist()
    label = label.to('cpu')
    for i in label:
        total_label.append(i)
    for i in outputs:
        total_outputs.append(train_model.max_list(i))

real_label = []
predict_label = []

for label in total_label:
    real_label.append(label_list[int(label)])

for label in total_outputs:
    predict_label.append(label_list[int(label)])

result = classification_report(total_label, total_outputs, target_names=label_list)

print(result)