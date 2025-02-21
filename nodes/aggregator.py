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
from model import Classifier
from flask import Flask, request
from torch.utils.data import DataLoader


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class Aggregator:
    def __init__(self):
        self.cwd = os.path.dirname(__file__)
        self.global_model_path = os.path.join(
            self.cwd, "models", "global.pt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_classes = 10

    def start(self, submits, prev_price, scale, total_rewards):
        self.combine_test_data(submits)
        submits = self.compute_contribution(submits)
        self.update_global_model(submits)
        self.update_personalized_models(submits)
        new_model_price = self.get_model_price(prev_price, scale)
        submits = self.compute_rewards(submits, total_rewards)
        self.store_on_ledger(new_model_price, submits)
        
    def store_on_ledger(self, new_model_price, submits):
        data = {
            "newPrice" : new_model_price,
            "submits" : []
        }
        for i in range(len(submits)):
            data["submits"].append({
                "walletId" : submits[i]["walletId"],
                "reward" : submits[i]["reward"],
                "modelPath" : submits[i]["model_path"]
            })
        requests.post("http://localhost:3000/api/aggregator/",
                      json=data)

    def compute_rewards(self, submits, total_rewards):
        contributions = [submit["contribution"] for submit in submits]
        reward_weights = np.array(contributions) / np.sum(contributions)
        for i in range(len(submits)):
            submits[i]["reward"] = (total_rewards * reward_weights[i]).item()
        return submits

    def get_model_price(self, prev_price, scale):
        self.g_model_loss = self.compute_loss(self.g_model)
        delta_loss = self.prev_g_model_loss - self.g_model_loss
        return prev_price + scale * delta_loss

    def get_normalization_factors(self, values):
        return np.array(values) / np.sum(values)

    def update_personalized_models(self, submits):
        gammas = self.get_normalization_factors(
            [submit["contribution"] for submit in submits])
        for i in range(len(submits)):
            model_state = submits[i]["model"].state_dict()
            model_params = {key: torch.zeros_like(
                value, dtype=torch.float32, device="cpu") for key, value in model_state.items()}
            g_model_state = self.g_model.state_dict()
            with torch.no_grad():
                for key in model_params:
                    model_params[key] = (
                        1 - gammas[i]) * model_state[key].cpu() + gammas[i] * g_model_state[key].cpu()
            submits[i]["model"].load_state_dict(model_params)
            model_path = os.path.join(
                self.cwd, "models", f"g_model_{i}.pt")
            torch.save(submits[i]["model"].state_dict(), model_path)
            submits[i]["model_path"] = model_path

    def softmax(self, values):
        return np.exp(values) / np.sum(np.exp(values))

    def update_global_model(self, submits):
        self.g_model = Classifier(self.num_classes).to(self.device)
        weights = self.softmax([submit["contribution"] for submit in submits])

        model_state = self.g_model.state_dict()
        model_params = {key: torch.zeros_like(
            value, dtype=torch.float32, device="cpu") for key, value in model_state.items()}
        with torch.no_grad():
            for i, submit in enumerate(submits):
                submit_model_state = submit["model"].state_dict()
                for key in model_params:
                    model_params[key] += weights[i] * \
                        submit_model_state[key].cpu()

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
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=len(self.test_dataset))

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
    # executer.submit(aggregator.start, submits,
    #                 prev_price, scale, total_rewards)
    return "aggregation started."


@app.route("/exit/")
def exit_aggregator():
    os.kill(os.getpid(), signal.SIGTERM)


if __name__ == '__main__':
    app.run(host="localhost", port=port)
