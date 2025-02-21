import os
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader

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


class Aggregator:
    def __init__(self):
        # These should be passed to the aggregator
        self.cwd = os.path.dirname(__file__)
        self.global_model_path = os.path.join(self.cwd, "nodes", "models", "global.pt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss_fn = nn.CrossEntropyLoss()
        self.prev_price = 50
        self.scale = 20
        self.total_rewards = 300
        self.num_classes = 10
        

    def start(self, submits):
        """
        submit = {id, path, testDataPath}
        """
        self.combine_test_data(submits)
        submits = self.compute_contribution(submits)
        self.update_global_model(submits)
        self.update_personalized_models(submits)
        new_model_price = self.get_model_price()
        submits = self.compute_rewards(submits)
        print(new_model_price)
        print([submit["reward"] for submit in submits])
        # Store rewards and new price on the ledger.


    def compute_rewards(self, submits):
        contributions = [submit["contribution"] for submit in submits]
        reward_weights = np.array(contributions) / np.sum(contributions)
        for i in range(len(submits)):
            submits[i]["reward"] = self.total_rewards * reward_weights[i]
        return submits


    def get_model_price(self):
        self.g_model_loss = self.compute_loss(self.g_model)
        delta_loss = self.g_model_loss - self.prev_g_model_loss
        return self.prev_price + self.scale * delta_loss


    def get_normalization_factors(self, values):
        return np.array(values) / np.sum(values)


    def update_personalized_models(self, submits):
        gammas = self.get_normalization_factors(
            [submit["contribution"] for submit in submits])
        for i in range(len(submits)):
            model_state = submits[i]["model"].state_dict()
            model_params = {key: torch.zeros_like(value, dtype=torch.float32, device="cpu") for key, value in model_state.items()}
            g_model_state = self.g_model.state_dict()  
            with torch.no_grad():
                for key in model_params:
                    model_params[key] = (1 - gammas[i]) * model_state[key].cpu() + gammas[i] * g_model_state[key].cpu()
            submits[i]["model"].load_state_dict(model_params)
            model_path = os.path.join(self.cwd, "nodes", "models", f"g_model_{i}.pt")
            torch.save(submits[i]["model"].state_dict(), model_path)
            submits[i]["model_path"] = model_path


    def softmax(self, values):
        return np.exp(values) / np.sum(np.exp(values))


    def update_global_model(self, submits):
        self.g_model = Classifier(self.num_classes).to(self.device)
        weights = self.softmax([submit["contribution"] for submit in submits])

        model_state = self.g_model.state_dict()
        model_params = {key: torch.zeros_like(value, dtype=torch.float32, device="cpu") for key, value in model_state.items()}  
        with torch.no_grad():
            for i, submit in enumerate(submits):
                submit_model_state = submit["model"].state_dict()
                for key in model_params:
                    model_params[key] += weights[i] * submit_model_state[key].cpu()
        
        self.g_model.load_state_dict(model_params)
        torch.save(self.g_model.state_dict(), self.global_model_path)



    def compute_contribution(self, submits):
        self.prev_g_model = self.load_model(self.global_model_path)
        self.prev_g_model_loss = self.compute_loss(self.prev_g_model)
        for submit in submits:
            submit["model"] = self.load_model(submit["path"])
            loss = self.compute_loss(submit["model"])
            delta = self.prev_g_model_loss - loss
            submit["contribution"] = max(delta, 0)
        return submits


    def combine_test_data(self, submits):
        datasets = []
        for submit in submits:
            dataset = torch.load(submit["testDataPath"])
            datasets.append(dataset)
        self.test_dataset = torch.utils.data.ConcatDataset(datasets)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=len(self.test_dataset))


    def load_model(self, path):
        model = Classifier(self.num_classes).to(self.device)
        model.load_state_dict(torch.load(path))
        return model


    def compute_loss(self, model):
        model.eval()
        with torch.no_grad():
            for X, y in self.test_dataloader:
                X = X.to(self.device)
                outputs = model(X)
                loss = self.loss_fn(outputs, y)
        return loss.item()


submits = [
    {
        "id": "model_0",
        "path": "/Users/amirrezasokhankhosh/Documents/Workstation/perfed/nodes/models/model_0.pt",
        "testDataPath": "/Users/amirrezasokhankhosh/Documents/Workstation/perfed/nodes/tests/tests_0.pt",
        "walletId": "wallet_0"
    },
    {
        "id": "model_1",
        "path": "/Users/amirrezasokhankhosh/Documents/Workstation/perfed/nodes/models/model_1.pt",
        "testDataPath": "/Users/amirrezasokhankhosh/Documents/Workstation/perfed/nodes/tests/tests_1.pt",
        "walletId": "wallet_1"
    },
    {
        "id": "model_2",
        "path": "/Users/amirrezasokhankhosh/Documents/Workstation/perfed/nodes/models/model_2.pt",
        "testDataPath": "/Users/amirrezasokhankhosh/Documents/Workstation/perfed/nodes/tests/tests_2.pt",
        "walletId": "wallet_2"
    },
    {
        "id": "model_3",
        "path": "/Users/amirrezasokhankhosh/Documents/Workstation/perfed/nodes/models/model_3.pt",
        "testDataPath": "/Users/amirrezasokhankhosh/Documents/Workstation/perfed/nodes/tests/tests_3.pt",
        "walletId": "wallet_3"
    }
]

aggregator = Aggregator()
aggregator.start(submits)