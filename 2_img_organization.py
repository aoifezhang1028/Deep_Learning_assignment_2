import os
from shutil import copyfile

classification_img_path = 'classification_img_CIFAR'
label_list = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# make dir
for label in label_list:
    label_folder = os.path.join(classification_img_path, 'train', label)
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)
    label_folder = os.path.join(classification_img_path, 'test', label)
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)

print('Dir done !')
# training organization

img_list = os.listdir('train')
for img in img_list:
    img_class = int(img[0])  # label img class
    start = os.path.join('train', img)
    end = os.path.join(classification_img_path, 'train', label_list[img_class], img)
    copyfile(start, end)  # move

# test organization
img_list = os.listdir('test')
for img in img_list:
    img_class = int(img[0])  # label img class
    start = os.path.join('test', img)
    end = os.path.join(classification_img_path, 'test', label_list[img_class], img)
    copyfile(start, end)  # move

print('done !!!')