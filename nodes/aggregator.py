import os
import sys
import json
import torch
import signal
import requests
import threading
import concurrent.futures
from flask import Flask, request


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


port = 8080
GLOBAL_MODEL_PATH = ""
executer = concurrent.futures.ThreadPoolExecutor(2)
app = Flask(__name__)


def start(submits, prev_price, scale, total_rewards):
    """
    submit = {id, path, testDataPath}
    """
    X, y = combine_test_data(submits)
    prev_g_model = load_model(GLOBAL_MODEL_PATH)
    prev_g_model_loss = get_loss(prev_g_model, X, y)
    submits = compute_contribution(submits, prev_g_model_loss, X, y)
    g_model = update_global_model(submits)
    g_model_loss = get_loss(g_model, X, y)
    update_personalized_models(submits)
    new_model_price = get_model_price(
        g_model_loss, prev_g_model_loss, prev_price, scale, X, y)
    submits = compute_rewards(submits, total_rewards)
    # Store reward and new price on the ledger.


def compute_rewards(submits, total_rewards):
    contributions = [submit["contribution"] for submit in submits]
    reward_weights = contributions / sum(contributions)
    for i in range(submits):
        submits[i]["reward"] = total_rewards * reward_weights[i]
    return submits


def get_model_price(g_model_loss, prev_g_model_loss, prev_price, scale, X, y):
    delta_loss = g_model_loss - prev_g_model_loss
    return prev_price + scale * delta_loss


def get_normalization_factors(values):
    pass


def update_personalized_models(submits):
    gammas = get_normalization_factors(
        [submit["contribution"] for submit in submits])
    # update models based on gammas


def softmax(values):
    pass


def update_global_model(submits):
    weights = softmax([submit["contribution"] for submit in submits])
    # update model using weighted sum.


def compute_contribution(submits, prev_g_model_loss, X, y):
    for submit in submits:
        model = load_model(submit["path"])
        loss = get_loss(model, X, y)
        delta = prev_g_model_loss - loss
        submit["contribution"] = max(delta, 0)
    return submits


def combine_test_data(submits):
    pass


def load_model(path):
    pass


def get_loss(model, X, y):
    pass


@app.route("/aggregate/")
def start_aggregate():
    submits = request.get_json()["submits"]
    executer.submit(start, submits)


@app.route("/exit/")
def exit_aggregator():
    os.kill(os.getpid(), signal.SIGTERM)


if __name__ == '__main__':
    app.run(host="localhost", port=port)
