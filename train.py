import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F
import pandas as pd
from sklearn import preprocessing


class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 100)
        self.layer2 = nn.Linear(100, 60)
        self.layer3 = nn.Linear(60, 5)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x))  # To check with the loss function
        return x


def load_dataset_files():
    """ Load shuffled, splitted dev and train files from .csv files. """

    cars_dev = pd.read_csv(f'data/Car_Prices_Poland_Kaggle_dev.csv', usecols=[1, 4, 5, 6, 10], sep=',', names= [str(i) for i in range(5)])
    cars_train = pd.read_csv(f'data/Car_Prices_Poland_Kaggle_train.csv', usecols=[1, 4, 5, 6, 10], sep=',', names= [str(i) for i in range(5)])

    return cars_dev, cars_train


def remove_rows(data_dev, data_train):
    dev_removed_rows = data_dev.loc[(data_dev['0'] == 'audi') | (data_dev['0'] == 'bmw') | (data_dev['0'] == 'ford') | (data_dev['0'] == 'opel') | (data_dev['0'] == 'volkswagen')]
    train_removed_rows = data_train.loc[(data_train['0'] == 'audi') | (data_train['0'] == 'bmw') | (data_train['0'] == 'ford') | (data_train['0'] == 'opel') | (data_train['0'] == 'volkswagen')]

    return dev_removed_rows, train_removed_rows


def prepare_labels_features(dataset):
    """ Label make column"""
    le = preprocessing.LabelEncoder()
    mark_column = np.array(dataset[:]['0'])
    le.fit(mark_column)

    print(list(le.classes_))
    lab = le.transform(mark_column)
    feat = dataset.drop(['0'], axis=1).to_numpy()

    mm_scaler = preprocessing.MinMaxScaler()
    feat = mm_scaler.fit_transform(feat)

    return lab, feat


if __name__ == "__main__":
    # Prepare dataset
    print("Loading dataset...")
    dev, train = load_dataset_files()
    print("Dataset loaded")

    print("Preparing dataset...")
    dev, train = remove_rows(dev, train)
    labels_train, features_train = prepare_labels_features(train)
    labels_test, features_test = prepare_labels_features(dev)
    print("Dataset prepared")

    print("Training...")
    model = Model(features_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 100
    print(f"Number of epochs: {epochs}")

    print("Starting model training...")
    x_train, y_train = Variable(torch.from_numpy(features_train)).float(), Variable(torch.from_numpy(labels_train)).long()
    for epoch in range(1, epochs + 1):
        print("Epoch #", epoch)
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        print(f"The loss calculated: {loss}")

        # Zero gradients
        optimizer.zero_grad()
        loss.backward()  # Gradients
        optimizer.step()  # Update
    print("Model training finished")

    print("Predictions...")
    x_test = Variable(torch.from_numpy(features_test)).float()
    pred = model(x_test)
    pred = pred.detach().numpy()
    acc = accuracy_score(labels_test, np.argmax(pred, axis=1))
    f1 = f1_score(labels_test, np.argmax(pred, axis=1), average='weighted')
    print(f"The accuracy metric is: {acc}")
    print(f"The f1 metric is: {f1}")

    with open("metrics.txt", 'w') as outfile:
        outfile.write("Accuracy: " + str(acc) + "\n")
        outfile.write("F1: " + str(f1) + "\n")

