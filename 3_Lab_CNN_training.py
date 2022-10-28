import torch
import torchvision.transforms
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import accuracy_score
import numpy as np
import train_model
import torch.nn as nn
from sklearn.metrics import classification_report

# Hyper Parameters
EPOCH = 50
BATCH_SIZE = 8
model = 'resnet'  # vgg, resnet
LR = 0.001
test_epoch = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
schedular_step = [10, 30]
save_folder = './classification_lab'
label_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


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
        img = train_model.RGB2Lab(img_path)
        img_label = self.label_dir
        # img transform
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


if __name__ == '__main__':
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_id = len(os.listdir(save_folder))
    save_folder = str(save_folder) + '/run_' + str(save_id)
    os.makedirs(save_folder)
    writer = SummaryWriter(str(save_folder) + '/classification_logs_Lab')

    # writing the hyperparameter to txt
    f = open(str(save_folder) + "/hyperparameter.txt", 'a')
    f.write('Epoch: ' + str(EPOCH))
    f.write('\n')
    f.write('BATCH_SIZE: ' + str(BATCH_SIZE))
    f.write('\n')
    f.write('Model: ' + str(model))
    f.write('\n')
    f.write('LR: ' + str(LR))
    f.write('\n')
    f.write('Test_epoch: ' + str(test_epoch))
    f.write('\n')
    f.write('Schedular_step: ' + str(schedular_step))
    f.close()

    # model loading
    if model == 'vgg':
        model_Yu = train_model.cifar10_VGG()

    elif model == 'resnet':
        model_Yu = train_model.cifar10_Resnet()

    model_Yu.to(device)

    # network parameter
    optimizer = torch.optim.Adam(model_Yu.parameters(), lr=LR)
    # optimizer = torch.optim.SGD(model_Yu.parameters(), lr=LR, momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=schedular_step, gamma=0.1)
    loss_func = nn.CrossEntropyLoss()
    loss_func.to(device)

    # training dataset
    root_dir_train = 'classification_img_CIFAR/train'

    airplane_dataset_train = img_dataset(root_dir_train, 'airplane')
    full_dataset_train = airplane_dataset_train

    for label in label_list[1:]:
        full_dataset_train = full_dataset_train + img_dataset(root_dir_train, str(label))

    # test dataset
    root_dir_test = 'classification_img_CIFAR/test'

    airplane_dataset_test = img_dataset(root_dir_test, 'airplane')
    full_dataset_test = airplane_dataset_test

    for label in label_list[1:]:
        full_dataset_test = full_dataset_test + img_dataset(root_dir_test, str(label))

    # loading data
    train_dataloader = DataLoader(full_dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(full_dataset_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    print(device)

    # initial F1
    last_F1 = 0

    for epoch in range(EPOCH):
        # train section
        total_loss_train = 0
        model_Yu.train(True)
        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)
            output = model_Yu(x)
            loss = loss_func(output, y)
            total_loss_train += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # tensorboard
        train_loss = float(total_loss_train.item() / (60000 / int(BATCH_SIZE)))
        writer.add_scalar(tag='Train loss', scalar_value=float(train_loss), global_step=int(epoch))
        # train section loss print
        if epoch % test_epoch == 0:
            print('---------- Train loss ----------')
            print('Epoch: ' + str(epoch) + ' learning rate: ' + str(optimizer.state_dict()['param_groups'][0]['lr']))
            print('Epoch: ' + str(epoch) + ' train loss: ' + str(train_loss))

        # test section
        if epoch % test_epoch == 0:
            model_Yu.eval()
            with torch.no_grad():
                total_loss = 0
                total_label = []
                total_outputs = []
                for data in test_dataloader:
                    img, label = data
                    img = img.to(device)
                    label = label.to(device)
                    outputs = model_Yu(img)
                    loss_test = loss_func(outputs, label)
                    total_loss = total_loss + loss_test
                    outputs = outputs.tolist()
                    label = label.to('cpu')
                    for i in label:
                        total_label.append(i)
                    for i in outputs:
                        total_outputs.append(train_model.max_list(i))

                # print result
                print('---------- Test loss ----------')
                test_loss = float(total_loss_train.item()/(10000/int(BATCH_SIZE)))
                print('Epoch: ' + str(epoch) + ' test loss: ' + str(test_loss))
                # tensorboard
                writer.add_scalar(tag='Test loss', scalar_value=float(test_loss), global_step=int(epoch))
                total_label = np.array(total_label)
                total_outputs = np.array(total_outputs)

                # tensorboard
                # Accuracy
                accuracy = accuracy_score(total_label, total_outputs)
                print('Epoch: ', str(epoch), ' Accuracy: ', round(accuracy, 3))
                writer.add_scalar(tag='Accuracy:', scalar_value=accuracy, global_step=epoch)

                # F1
                F1 = f1_score(total_label, total_outputs, average='weighted')
                print('Epoch: ', str(epoch), ' F1 score: ', round(F1, 3))
                writer.add_scalar(tag='F1 score', scalar_value=F1, global_step=epoch)

            current_F1 = round(F1, 3)
            # best model pth save
            if current_F1 > last_F1:
                print('Last F1: ', str(last_F1), ' Current F1: ' + str(current_F1))
                result = classification_report(total_label, total_outputs, target_names=label_list)
                print(result)
                save_best_model = str(current_F1) + '_' + str(model) + '_best.pth'
                torch.save(model_Yu, str(save_folder) + '/' + save_best_model)
                # torch.save(model_Yu, str(save_folder) + '/best.pth')
                last_F1 = current_F1

            # last model pth save
            torch.save(model_Yu, str(save_folder) + '/last.pth')
            scheduler.step()

    writer.close()