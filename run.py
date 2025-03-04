import os
import time
import requests
import subprocess

def stop_nodes(num_nodes):
    try:
        requests.get("http://localhost:3000/exit/")
    except:
        print("Express app is stopped.")

    try:
        requests.get("http://localhost:8080/exit/")
    except:
        print("Aggregator is stopped.")

    # Nodes
    for i in range(num_nodes):
        try:
            requests.get(f"http://localhost:{8000 + i}/exit/")
        except:
            print(f"Node{i} is stopped.")

def run_node(i):
    log_file = f"../logs/node_{i}.txt"
    with open(log_file, "w") as f:
        subprocess.Popen(
            ["python3", f"node{i}.py"],
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            preexec_fn=os.setsid
        )

if __name__ == "__main__":
    num_nodes = 4
    cwd = os.path.dirname(__file__)

    stop_nodes(num_nodes)

    # Step 1: Bring up the network
    os.chdir(os.path.join(cwd, "test-network"))
    os.system("sh ./start.sh")

    # Step 2: Bring up the express app
    os.chdir(os.path.join(cwd, "express-application"))
    with open("../logs/app.txt", "w") as f:
        subprocess.Popen(
            ["node", f"./app.js"],
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            preexec_fn=os.setsid  # To run the process in a new session (for Unix-like systems)
        )

    os.chdir(os.path.join(cwd, "nodes"))
    os.system("python3 model.py")

    print("Bringing up the nodes...")
    os.chdir(os.path.join(cwd, "nodes"))
    for i in range(num_nodes):
        run_node(i)
        time.sleep(1)
    
    print("Bringing up the aggregator")
    os.chdir(os.path.join(cwd, "nodes"))
    with open("../logs/aggregator.txt", "w") as f:
        subprocess.Popen(
            ["python3", f"aggregator.py"],
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            preexec_fn=os.setsid
        )
    
    # Step 3: Initialize wallets
    os.chdir(os.path.join(cwd, "test-network"))
    os.system("sh ./req.sh")
    
    path = os.path.join(cwd, "nodes", "models", "global.pt")
    print("\nStarting the training process...")
    time.sleep(5)
    requests.post("http://localhost:3000/api/start/", 
        json = {
            "globalModelPath" : path
        }              
    )