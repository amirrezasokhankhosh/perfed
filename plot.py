import matplotlib.pyplot as plt
import json
import numpy as np

with open('./nodes/results/res.json', 'r') as f:
    data = json.load(f)

rounds = []
global_losses = []
new_model_prices = []

wallets = {}

for round_data in data:
    rounds.append(round_data['round'])
    global_losses.append(round_data['g_model_loss'])
    new_model_prices.append(round_data['new_model_price'])
    
    for submit in round_data['submits']:
        wid = submit['walletId']
        if wid not in wallets:
            wallets[wid] = {
                'loss': [],
                'delta_local_loss': [],
                'delta_gap': [],
                'contribution': [],
                'reward': []
            }
        wallets[wid]['loss'].append(submit['loss'])
        wallets[wid]['delta_local_loss'].append(submit['delta_local_loss'])
        wallets[wid]['delta_gap'].append(submit['delta_gap'])
        wallets[wid]['contribution'].append(submit['contribution'])
        wallets[wid]['reward'].append(submit['reward'])

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(rounds, global_losses, marker='o', linestyle='-', color='blue', label='Global Loss')
plt.xlabel('Round')
plt.ylabel('Global Model Loss')
plt.title('Global Model Loss Over Rounds')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(rounds, new_model_prices, marker='s', linestyle='-', color='orange', label='New Model Price')
plt.xlabel('Round')
plt.ylabel('New Model Price')
plt.title('New Model Price Over Rounds')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("./figures/g_model_loss.png", dpi=500)

plt.figure(figsize=(12, 6))
for wid, metrics in wallets.items():
    plt.plot(rounds, metrics['contribution'], marker='x', linestyle='-', label=f'{wid} Contribution')
plt.xlabel('Round')
plt.ylabel('Contribution')
plt.title('Wallet Contributions Over Rounds')
plt.grid(True)
plt.legend()
plt.savefig("./figures/contribution.png", dpi=500)

plt.figure(figsize=(12, 6))
for wid, metrics in wallets.items():
    plt.plot(rounds, metrics['reward'], marker='^', linestyle='-', label=f'{wid} Reward')
plt.xlabel('Round')
plt.ylabel('Reward')
plt.title('Wallet Rewards Over Rounds')
plt.grid(True)
plt.legend()
plt.savefig("./figures/rewards.png", dpi=500)

plt.figure(figsize=(12, 6))
for wid, metrics in wallets.items():
    plt.plot(rounds, metrics['loss'], marker='o', linestyle='-', label=f'{wid} Local Loss')
plt.xlabel('Round')
plt.ylabel('Local Loss')
plt.title('Wallet Local Loss Over Rounds')
plt.grid(True)
plt.legend()
plt.savefig("./figures/local_losses.png", dpi=500)

plt.figure(figsize=(12, 6))
for wid, metrics in wallets.items():
    plt.plot(rounds, metrics['delta_local_loss'], marker='o', linestyle='-', label=f'{wid} Delta Local Loss')
plt.xlabel('Round')
plt.ylabel('Delta Local Loss')
plt.title('Delta Local Loss Over Rounds')
plt.grid(True)
plt.legend()
plt.savefig("./figures/delta_local_losses.png", dpi=500)

plt.figure(figsize=(12, 6))
for wid, metrics in wallets.items():
    plt.plot(rounds, metrics['delta_gap'], marker='o', linestyle='-', label=f'{wid} Delta Gap')
plt.xlabel('Round')
plt.ylabel('Delta Gap')
plt.title('Delta Gap Over Rounds')
plt.grid(True)
plt.legend()
plt.savefig("./figures/delta_gap.png", dpi=500)
