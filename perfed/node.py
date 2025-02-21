import os
import json
import torch
import random
import requests
import numpy as np
from torch import nn
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 4 * 4, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.mp(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.mp(x)

        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.mp(x)

        x = self.flatten(x)
        x = self.dropout(x)
        return self.softmax(self.fc(x))


class Node:
    def __init__(self, num_nodes, port, peer_port=3000):
        self.num_nodes = num_nodes
        self.random_state = 1379
        self.peer_port = peer_port
        self.index = port - 8000

        cwd = os.path.dirname(__file__)
        self.model_path = os.path.abspath(os.path.join(
            cwd, "..", "nodes", "models", f"model_{self.index}.pt"))
        self.tests_path = os.path.abspath(os.path.join(
            cwd, "..", "nodes", "tests", f"tests_{self.index}.pt"))

        self.round = 0
        self.epochs = 1
        self.batch_size = 32

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Classifier(num_classes=10).to(self.device)
        self.get_data()
        self.save_random_tests()

    def get_data(self):
        train_dataset = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=ToTensor()
        )
        test_dataset = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )

        data_portion = len(train_dataset) // self.num_nodes
        indexes = [random.randint(0, len(train_dataset) - 1)
                   for _ in range(data_portion)]
        test_portion = len(test_dataset) // self.num_nodes
        test_indexes = [random.randint(0, len(test_dataset) - 1)
                        for _ in range(test_portion)]

        self.train_dataset = torch.utils.data.Subset(train_dataset, indexes)
        self.test_dataset = torch.utils.data.Subset(test_dataset, test_indexes)

        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size)
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=test_portion)
    
    def save_random_tests(self):
        test_portion = len(self.test_dataset) // self.num_nodes
        test_indexes = [random.randint(0, len(self.test_dataset) - 1)
                        for _ in range(test_portion)]
        selected_tests = torch.utils.data.Subset(self.test_dataset, test_indexes)
        torch.save(selected_tests, self.tests_path)

    def train(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters())
        loss_fn = nn.CrossEntropyLoss()
        for e in range(self.epochs):
            print(f"Epoch {e+1}:")
            for (X, y) in tqdm(iter(self.train_dataloader)):
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss = loss_fn(pred, y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        torch.save(self.model.state_dict(), self.model_path)
        requests.post("http://localhost:3000/api/model/",
                      json={
                          "id": f"model_{self.index}",
                          "walletId": f"wallet_{self.index}",
                          "path": f"{self.model_path}",
                          "testDataPath" : f"{self.tests_path}"
                      })
        self.round += 1

    def predict(self):
        pass
