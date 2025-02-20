import requests

try:
    requests.get("http://localhost:3000/exit/")
except:
    print("App1 is stopped.")

# Nodes
for i in range(4):
    try:
        requests.get(f"http://localhost:{8000 + i}/exit/")
    except:
        print(f"Node{i} is stopped.")