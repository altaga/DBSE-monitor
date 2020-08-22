import torch
import torch.nn as nn
import torchvision.transforms as transforms
import random
import model
import glob
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

shape = (24, 24)
validation_ratio = 0.1
lr = 0.001
batch_size = 64
epochs = 12


def resize(image, bbox):
    (x,y,w,h) = bbox
    eye = image[y:y + h, x:x + w]
    return Image.fromarray(cv2.resize(eye, shape))


class DataSetFactory:

    def __init__(self):
        images = []
        labels = []

        files = list(map(lambda x: {'file': x, 'label':1}, glob.glob('dataset/dataset_B_Eye_Images/openRightEyes/*.jpg')))
        files.extend(list(map(lambda x: {'file': x, 'label':1}, glob.glob('dataset/dataset_B_Eye_Images/openLeftEyes/*.jpg'))))
        files.extend(list(map(lambda x: {'file': x, 'label':0}, glob.glob('dataset/dataset_B_Eye_Images/closedLeftEyes/*.jpg'))))
        files.extend(list(map(lambda x: {'file': x, 'label':0}, glob.glob('dataset/dataset_B_Eye_Images/closedRightEyes/*.jpg'))))
        random.shuffle(files)
        for file in files:
            img = cv2.imread(file['file'])
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            labels.append(file['label'])

        validation_length = int(len(images) * validation_ratio)
        validation_images = images[:validation_length]
        validation_labels = labels[:validation_length]
        images = images[validation_length:]
        labels = labels[validation_length:]

        print('training size %d : val size %d' % (len(images), len(validation_images)))

        train_transform = transforms.Compose([
            ToTensor(),
        ])
        val_transform = transforms.Compose([
            ToTensor(),
        ])

        self.training = DataSet(transform=train_transform, images=images, labels=labels)
        self.validation = DataSet(transform=val_transform, images=validation_images, labels=validation_labels)


class DataSet(torch.utils.data.Dataset):

    def __init__(self, transform=None, images=None, labels=None):
        self.transform = transform
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.images)



def main():
    global batch_size
    global epochs
    global lr
    
    # ------------------------

    factory = DataSetFactory()
    training_loader = DataLoader(factory.training, batch_size=batch_size, shuffle=True, num_workers=1)
    validation_loader = DataLoader(factory.validation, batch_size=batch_size, shuffle=True, num_workers=1)
    network = model.Model(num_classes=2).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    min_validation_loss = 10000
    for epoch in range(epochs):
        network.train()
        total = 0
        correct = 0
        total_train_loss = 0

        for i, (x_train, y_train) in enumerate(training_loader):
            optimizer.zero_grad()
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            y_predicted = network(x_train)
            loss = criterion(y_predicted, y_train)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(y_predicted.data, 1)
            total_train_loss += loss.data
            total += y_train.size(0)
            correct += predicted.eq(y_train.data).sum()
        accuracy = 100. * float(correct) / total
        print('Epoch [%d/%d] Training Loss: %.4f, Accuracy: %.4f' % (
            epoch + 1, epochs, total_train_loss / (i + 1), accuracy))

        network.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            total_validation_loss = 0
            for j, (x_val, y_val) in enumerate(validation_loader):
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                y_val_predicted = network(x_val)
                val_loss = criterion(y_val_predicted, y_val)
                _, predicted = torch.max(y_val_predicted.data, 1)
                total_validation_loss += val_loss.data
                total += y_val.size(0)
                correct += predicted.eq(y_val.data).sum()

            accuracy = 100. * float(correct) / total
            if total_validation_loss <= min_validation_loss:
                if epoch >= 10:
                    print('saving new model')
                    state = {'net': network.state_dict()}
                    torch.save(state, 'Model/model_%d_%d.t7' % (epoch, batch_size))
                    min_validation_loss = total_validation_loss

            print('Epoch [%d/%d] validation Loss: %.4f, Accuracy: %.4f' % (
                epoch + 1, epochs, total_validation_loss / (j + 1), accuracy))


if __name__ == "__main__":
    main()
