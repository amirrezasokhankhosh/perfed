import os
import sys
import json
import torch
import signal
import requests
import threading
import numpy as np
from torch import nn
import concurrent.futures
from model import ResNet18Classifier
from flask import Flask, request
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class Aggregator:
    def __init__(self):
        self.cwd = os.path.dirname(__file__)
        self.global_model_path = os.path.join(self.cwd, "models", "global.pt")
        self.device = "cpu"
        self.loss_fn = nn.CrossEntropyLoss()
        self.gamma_max = 0.7
        self.p = 0.5
        self.contribution_factor = 0.5
        self.num_classes = 10
        self.curr_round = 0
        self.results = {}
        self.prev_losses = {}

    def start(self, submits, prev_price, scale, total_rewards):
        self.combine_test_data(submits)
        submits = self.compute_contribution(submits)
        self.update_global_model(submits)
        self.update_personalized_models(submits)
        new_model_price = self.get_model_price(prev_price, scale)
        submits = self.compute_rewards(submits, total_rewards)
        self.store_on_ledger(new_model_price, submits)
        self.save_results(new_model_price, submits)
        self.save_losses(submits)

    def save_losses(self, submits):
        for submit in submits:
            self.prev_losses[submit["walletId"]] = submit["loss"]
    
    def save_results(self, new_model_price, submits):
        self.curr_round += 1
        data = {
            "round": self.curr_round,
            "new_model_price": new_model_price,
            "g_model_loss": self.g_model_loss,
            "submits": []
        }
        for submit in submits:
            data["submits"].append({
                "walletId": submit["walletId"],
                "loss": submit["loss"],
                "delta_local_loss" : submit["delta_local_loss"],
                "delta_gap" : submit["delta_gap"],
                "contribution": submit["contribution"],
                "reward": submit["reward"],
            })
        if self.curr_round == 1:
            self.results = []
        self.results.append(data)
        with open("./results/res.json", "w") as f:
            f.write(json.dumps(self.results, indent=2))
            
    def store_on_ledger(self, new_model_price, submits):
        data = {
            "newPrice": new_model_price,
            "submits": []
        }
        for submit in submits:
            data["submits"].append({
                "walletId": submit["walletId"],
                "reward": submit["reward"],
                "modelPath": submit["model_path"]
            })
        requests.post("http://localhost:3000/api/aggregator/", json=data)

    def compute_rewards(self, submits, total_rewards):
        contributions = np.array([submit["contribution"] for submit in submits])
        smoothed = np.log1p(contributions)
        sum_smoothed = np.sum(smoothed)
        if sum_smoothed == 0 or np.isnan(sum_smoothed) or np.isinf(sum_smoothed):
            reward_weights = np.ones_like(contributions) / len(contributions)
        else:
            reward_weights = smoothed / sum_smoothed
        for i in range(len(submits)):
            submits[i]["reward"] = (total_rewards * reward_weights[i]).item()
        return submits

    def get_model_price(self, prev_price, scale):
        self.g_model_loss = self.compute_loss(self.g_model)
        delta_loss = self.prev_g_model_loss - self.g_model_loss
        return prev_price + scale * delta_loss

    def get_gammas(self, values):
        values = np.array(values, dtype=np.float64)
        max_val = np.max(values)
        if max_val == 0 or np.isnan(max_val) or np.isinf(max_val):
            return np.ones_like(values)
        return np.minimum(np.power((values / max_val), self.p), self.gamma_max)

    def update_personalized_models(self, submits):
        gammas = self.get_gammas([submit["contribution"] for submit in submits])
        for i in range(len(submits)):
            model_state = submits[i]["model"].state_dict()
            model_params = {key: torch.zeros_like(value, dtype=torch.float32, device="cpu") 
                            for key, value in model_state.items()}
            g_model_state = self.g_model.state_dict()
            with torch.no_grad():
                for key in model_params:
                    model_params[key] = ((1 - gammas[i]) * model_state[key].cpu() +
                                         gammas[i] * g_model_state[key].cpu())
            submits[i]["model"].load_state_dict(model_params)
            model_path = os.path.join(self.cwd, "models", f"g_model_{i}.pt")
            torch.save(submits[i]["model"].state_dict(), model_path)
            submits[i]["model_path"] = model_path

    def softmax(self, values):
        values = np.array(values, dtype=np.float64)
        max_val = np.max(values)
        if max_val == 0 or np.isnan(max_val) or np.isinf(max_val):
            return np.ones_like(values) / len(values)
        values = values - max_val
        exp_values = np.exp(values)
        sum_exp = np.sum(exp_values) + 1e-9
        return exp_values / sum_exp

    def update_global_model(self, submits):
        self.g_model = ResNet18Classifier(self.num_classes).to(self.device)
        weights = self.softmax([submit["contribution"] for submit in submits])
        model_state = self.g_model.state_dict()
        model_params = {key: torch.zeros_like(value, dtype=torch.float32, device="cpu") 
                        for key, value in model_state.items()}
        with torch.no_grad():
            for i, submit in enumerate(submits):
                submit_model_state = submit["model"].state_dict()
                for key in model_params:
                    model_params[key] += weights[i] * submit_model_state[key]
        self.g_model.load_state_dict(model_params)
        torch.save(model_params, self.global_model_path)

    def compute_contribution(self, submits):
        self.prev_g_model = self.load_model(self.global_model_path)
        self.prev_g_model_loss = self.compute_loss(self.prev_g_model)
        for submit in submits:
            submit["model"] = self.load_model(submit["path"])
            loss = self.compute_loss(submit["model"])
            delta_gap = self.prev_g_model_loss - loss
            if self.curr_round != 0:
                delta_local_loss = self.prev_losses.get(submit["walletId"], self.prev_g_model_loss) - loss
                delta = self.contribution_factor * delta_gap + (1 - self.contribution_factor) * delta_local_loss
                submit["delta_local_loss"] = delta_local_loss
            else:
                submit["delta_local_loss"] = 0
                delta = delta_gap
            submit["loss"] = loss 
            submit["delta_gap"] = delta_gap
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
        model = ResNet18Classifier(self.num_classes).to(self.device)
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


port = 8080
executer = concurrent.futures.ThreadPoolExecutor(2)
aggregator = Aggregator()
app = Flask(__name__)


@app.route("/aggregate/", methods=['POST'])
def aggregate():
    submits = request.get_json()["submits"]
    prev_price = request.get_json()["price"]
    scale = request.get_json()["scale"]
    total_rewards = request.get_json()["totalRewards"]
    aggregator.start(submits, prev_price, scale, total_rewards)
    return "aggregation started."


@app.route("/exit/")
def exit_aggregator():
    os.kill(os.getpid(), signal.SIGTERM)


if __name__ == '__main__':
    app.run(host="localhost", port=port)
