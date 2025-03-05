import os
import json
import torch
import random
import requests
import numpy as np
import torchvision
from torch import nn
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2, ToTensor
from torch.optim.lr_scheduler import ReduceLROnPlateau

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class CustomDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


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
        self.data_path = os.path.abspath(os.path.join(
            cwd, "..", "data"))

        self.round = 0
        self.epochs = 5
        self.batch_size = 32
        self.lr = 1e-3
        self.wd = 1e-4

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ResNet18Classifier(num_classes=10).to(self.device)
        self.get_data()
        self.save_random_tests()

    def get_data(self):
        transforms = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(10),
            v2.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            v2.ToTensor()
        ])
        train_dataset = datasets.CIFAR10(
            root="/home/cs/grad/sokhanka/Documents/perfed/nodes/data",
            train=True,
            download=True
        )
        test_dataset = datasets.CIFAR10(
            root="/home/cs/grad/sokhanka/Documents/perfed/nodes/data",
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

        train_subset = torch.utils.data.Subset(train_dataset, indexes)
        self.train_dataset = CustomDataset(train_subset, transform=transforms)
        self.val_dataset = torch.utils.data.Subset(test_dataset, test_indexes[:test_portion//2])
        self.test_dataset = torch.utils.data.Subset(test_dataset, test_indexes[test_portion//2:])

        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size)
        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=test_portion//2)
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=test_portion//2)
    
    def save_random_tests(self):
        test_portion = len(self.test_dataset) // self.num_nodes
        test_indexes = [random.randint(0, len(self.test_dataset) - 1)
                        for _ in range(test_portion)]
        selected_tests = torch.utils.data.Subset(self.test_dataset, test_indexes)
        torch.save(selected_tests, self.tests_path)

    def train_epoch(self, loss_fn, optimizer):
        self.model.train()
        for (X, y) in iter(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)
            pred = self.model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def validate_epoch(self, loss_fn):
        self.model.eval()
        with torch.no_grad():
            X, y = next(iter(self.val_dataloader))
            X, y = X.to(self.device), y.to(self.device)
            pred = self.model(X)
            loss = loss_fn(pred, y)
        return loss.item()

    def train(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        loss_fn = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, "min", patience=self.epochs//6)
        for e in range(self.epochs):
            print(f"Epoch {e+1}:")
            self.train_epoch(loss_fn, optimizer)
            val_loss = self.validate_epoch(loss_fn)
            scheduler.step(val_loss)

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
