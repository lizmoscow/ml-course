import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, accuracy_score
import matplotlib.pyplot as plt


INPUTS_PER_USER_TRAIN = 400
INPUTS_PER_USER_VALID = 400
path = "DSL-StrongPasswordData.csv"
data = pd.read_csv(path)
subjects = data["subject"].unique()
subjects_train = subjects[:34]
subjects_test = subjects[34:]

train_set = pd.DataFrame(columns=data.columns)
valid_set = pd.DataFrame(columns=data.columns)
test_set = pd.DataFrame(columns=data.columns)
for s in subjects_train:
    train_set = train_set.append(data.loc[data.subject == s])
    valid_set = valid_set.append(data.loc[data.subject == s])

for s in subjects_test:
    test_set = test_set.append(data.loc[data.subject == s])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(31, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.4)
        self.sigmoid = nn.Sigmoid()

    def twin(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        return x

    def forward(self, x1, x2):
        x1 = self.twin(x1)
        x2 = self.twin(x2)
        x = torch.abs(x1 - x2)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc4(x))
        return x


class DatasetTrain(Dataset):
    def __init__(self, train_set, categories, length):
        self.categories = categories
        self.set = train_set
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input1 = None
        input2 = None
        label = None
        if idx % 2 == 0:
            s = random.choice(self.categories)
            rows = random.sample(range(INPUTS_PER_USER_TRAIN), 2)
            input1, input2 = self.set.loc[self.set.subject == s, "H.period":"H.Return"].iloc[rows].to_numpy(dtype=np.float32)
            label = 1.0
        else:
            s1, s2 = random.sample(self.categories.tolist(), 2)
            row1, row2 = random.sample(range(INPUTS_PER_USER_TRAIN), 2)
            input1 = self.set.loc[self.set.subject == s1, "H.period":"H.Return"].iloc[row1].to_numpy(dtype=np.float32)
            input2 = self.set.loc[self.set.subject == s2, "H.period":"H.Return"].iloc[row2].to_numpy(dtype=np.float32)
            label = 0.0
        return input1, input2, torch.from_numpy(np.array([label], dtype=np.float32))



def train(model, train_loader, val_loader, num_epochs, criterion, optimizer):
    train_losses = []
    val_losses = []
    cur_step = 0
    accuracies = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        print("Starting epoch " + str(epoch + 1))
        for img1, img2, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        val_running_loss = 0.0
        eer_av = 0
        lbls = []
        outpts = []
        with torch.no_grad():
            model.eval()
            for img1, img2, labels in val_loader:
                outputs = model(img1, img2)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                #eer_av += eer(labels, outputs)
                lbls.append(labels.tolist())
                outpts.append(outputs.tolist())
        lbls = np.array(lbls).squeeze()
        outpts = np.array(outpts).squeeze()
        avg_val_loss = val_running_loss / len(val_loader)
        eer_av = eer(lbls, outpts)
        outpts = outpts > eer_av
        outpts = outpts.astype(float)
        accuracy_av = accuracy_score(lbls, outpts)
        val_losses.append(avg_val_loss)
        accuracies.append(accuracy_av)
        print('Epoch [{}/{}],Train Loss: {:.4f}, Valid Loss: {:.8f}, EER: {}, Accuracy: {}'
              .format(epoch + 1, num_epochs, avg_train_loss, avg_val_loss, eer_av, accuracy_av))
        if epoch > 10:
            improvement = False
            for i in range(-10, 1):
                if accuracy_av < accuracies[epoch + i]:
                    improvement = True
            if improvement:
                for g in optimizer.param_groups:
                    g['lr'] *= 0.2
        if epoch > 30:
            stop = False
            for i in range(-30, 1):
                if accuracy_av < accuracies[epoch + i]:
                    return train_losses, val_losses
    print("Finished Training")
    return train_losses, val_losses



def eer(y, y_pred):
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
    fnr = 1 - tpr
    return threshold[np.nanargmin(np.absolute((fnr - fpr)))]


def eval(model, test_loader, threshold=0.48):
    with torch.no_grad():
        model.eval()
        lbls = []
        outpts = []

        for mainImg, imgSets, label in test_loader:
            output = model(mainImg, imgSets)
            lbls.append(label.tolist())
            outpts.append(output.tolist())
        lbls = np.array(lbls).squeeze()
        outpts = np.array(outpts).squeeze()
        eer_val = eer(np.array(lbls.tolist()).squeeze(), outpts)
        outputs = outpts > eer_val
        print('Accuracy for threshold = {}: {}'.format(eer_val, accuracy_score(lbls, outputs)))
        print('EER: {}'.format(eer_val))


model = Net()
train_loader = DataLoader(DatasetTrain(train_set, subjects_train, 75000), batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(DatasetTrain(train_set, subjects_train, 25000), batch_size=1, shuffle=True, num_workers=0)
num_epochs = 150
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)

# train(model, train_loader, val_loader, num_epochs, criterion, optimizer)
PATH = 'model2.pth'
# torch.save(model.state_dict(), PATH)
model.load_state_dict(torch.load(PATH))

test_loader = DataLoader(DatasetTrain(test_set, subjects_test, 25000), batch_size=1, shuffle=True, num_workers=0)
eval(model, test_loader)

