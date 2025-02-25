import os
import sys
import json
import signal
import requests
import threading
import concurrent.futures
from flask import Flask, request
from collections import defaultdict


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from perfed.node import Node

port = 8002
num_nodes = 4

executer = concurrent.futures.ThreadPoolExecutor(2)
node = Node(num_nodes=num_nodes, port=port)
app = Flask(__name__)

@app.route("/round/", methods=['POST'])
def train():
    path = request.get_json()["modelPath"]
    executer.submit(node.train, path)
    return "The node has started training."

@app.route("/exit/")
def exit_miner():
    os.kill(os.getpid(), signal.SIGTERM)
    
if __name__ == '__main__':
    app.run(host="localhost", port=port)