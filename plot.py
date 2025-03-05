
import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 12,
    'figure.figsize': (10, 6),
    'lines.linewidth': 2,
    'lines.markersize': 8
})

# Load the JSON results from file (adjust the filename if needed)
with open('./nodes/results/res.json', 'r') as f:
    data = json.load(f)

# Initialize lists for overall round metrics
rounds = []
global_losses = []
new_model_prices = []

# Initialize dictionaries to store per-wallet metrics
wallets = {}  # Keys will be wallet IDs, values will be dicts with lists for each metric

# Process each round in the JSON data
for round_data in data:
    rounds.append(round_data['round'])
    global_losses.append(round_data['g_model_loss'])
    new_model_prices.append(round_data['new_model_price'])
    
    # Iterate over each submit (client result) in this round
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

# Plot 1: Global Model Loss and New Model Price vs. Rounds
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs[0].plot(rounds, global_losses, marker='o', color='navy')
axs[0].set_xlabel('Round')
axs[0].set_ylabel('Global Model Loss')
axs[0].set_title('Global Model Loss Over Rounds')
axs[0].grid(True)

axs[1].plot(rounds, new_model_prices, marker='s', color='darkorange')
axs[1].set_xlabel('Round')
axs[1].set_ylabel('New Model Price')
axs[1].set_title('New Model Price Over Rounds')
axs[1].grid(True)

plt.tight_layout()
plt.savefig('./figures/global_metrics.png', dpi=300)

# Plot 2: Contributions for Each Wallet
plt.figure()
for wid, metrics in wallets.items():
    plt.plot(rounds, metrics['contribution'], marker='x', label=f'{wid}')
plt.xlabel('Round')
plt.ylabel('Contribution')
plt.title('Wallet Contributions Over Rounds')
plt.legend(title='Wallet ID')
plt.tight_layout()
plt.savefig('./figures/wallet_contributions.png', dpi=300)

# Plot 3: Rewards for Each Wallet
plt.figure()
for wid, metrics in wallets.items():
    plt.plot(rounds, metrics['reward'], marker='^', label=f'{wid}')
plt.xlabel('Round')
plt.ylabel('Reward')
plt.title('Wallet Rewards Over Rounds')
plt.legend(title='Wallet ID')
plt.tight_layout()
plt.savefig('./figures/wallet_rewards.png', dpi=300)

# Optional Plot 4: Local Loss for Each Wallet
plt.figure()
for wid, metrics in wallets.items():
    plt.plot(rounds, metrics['loss'], marker='o', label=f'{wid}')
plt.xlabel('Round')
plt.ylabel('Local Loss')
plt.title('Wallet Local Loss Over Rounds')
plt.legend(title='Wallet ID')
plt.tight_layout()
plt.savefig('./figures/wallet_local_loss.png', dpi=300)

# Optional Plot 5: Delta Local Loss per Wallet
plt.figure()
for wid, metrics in wallets.items():
    plt.plot(rounds[1:], metrics['delta_local_loss'][1:], marker='o', label=f'{wid}')
plt.xlabel('Round')
plt.ylabel('Delta Local Loss')
plt.title('Delta Local Loss Over Rounds')
plt.legend(title='Wallet ID')
plt.tight_layout()
plt.savefig('./figures/delta_local_loss.png', dpi=300)

# Optional Plot 6: Delta Gap per Wallet
plt.figure()
for wid, metrics in wallets.items():
    plt.plot(rounds, metrics['delta_gap'], marker='o', label=f'{wid}')
plt.xlabel('Round')
plt.ylabel('Delta Gap')
plt.title('Delta Gap Over Rounds')
plt.legend(title='Wallet ID')
plt.tight_layout()
plt.savefig('./figures/delta_gap.png', dpi=300)
