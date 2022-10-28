dataset_path = 'data/cifar-10-batches-py'

import imageio
import numpy as np
import pickle
import os

if not os.path.exists('train'):
    os.makedirs('train')
if not os.path.exists('test'):
    os.makedirs('test')

# unpickle the cifar_10_documents
def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict


# creat training image dataset
for j in range(1, 6):
    dataName = "data_batch_" + str(j)  # there are six documents for training
    dataName = os.path.join(dataset_path, dataName)
    Xtr = unpickle(dataName)
    print(dataName + " is processing")

    for i in range(0, 10000):
        img = np.reshape(Xtr['data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)  # loading image
        # labels -- a list of 10000 numbers in the range 0-9.
        # The number at index i indicates the label of the ith image in the array data.
        picName = 'train/' + str(Xtr['labels'][i]) + '_' + str(i + (j - 1)*10000) + '.jpg'
        imageio.imsave(picName, img)
    print(dataName + " loaded.")

print("Test_batch is processing...")

# creat test image dataset
testXtr = unpickle(os.path.join(dataset_path,'test_batch'))
for i in range(0, 10000):
    img = np.reshape(testXtr['data'][i], (3, 32, 32))
    img = img.transpose(1, 2, 0)
    picName = 'test/' + str(testXtr['labels'][i]) + '_' + str(i) + '.jpg'
    imageio.imsave(picName, img)
print("test_batch loaded.")

