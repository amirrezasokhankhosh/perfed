'use strict';

const stringify = require("json-stringify-deterministic");
const sortKeysRecursive = require("sort-keys-recursive");
const {Contract} = require("fabric-contract-api");

class TokenTransfer extends Contract {
    async InitWallets(ctx, numNodes, basePrice, scale, totalRewards) {
        for (let i = 0; i < parseInt(numNodes); i++) {
            const wallet = {
                id: `wallet_${i}`,
                balance: 0.0
            }
            await ctx.stub.putState(wallet.id, Buffer.from(stringify(sortKeysRecursive(wallet))));
        }
        const priceInfo = {
            id: "priceInfo",
            price: parseFloat(basePrice),
            scale: parseFloat(scale),
            totalRewards: parseFloat(totalRewards)
        }
        await ctx.stub.putState(priceInfo.id, Buffer.from(stringify(sortKeysRecursive(priceInfo))));
    }

    async KeyExists(ctx, id) {
        const valueBytes = await ctx.stub.getState(id);
        return valueBytes && valueBytes.length > 0;
    }

    async CreateWallet(ctx, id, balance) {
        const walletExists = await this.WalletExists(ctx, id);
        if (walletExists) {
            throw Error(`A wallet already exists with id: ${id}.`);
        }

        const wallet = {
            id: id,
            balance: parseFloat(balance)
        }

        await ctx.stub.putState(wallet.id, Buffer.from(stringify(sortKeysRecursive(wallet))));
    }

    async ReadKey(ctx, id) {
        const valueBytes = await ctx.stub.getState(id);
        if (!valueBytes || valueBytes.length === 0) {
            throw Error(`No wallet exists with id: ${id}.`);
        }
        return valueBytes.toString()
    }

    async GetAllWallets(ctx) {
        const allResults = [];
        const iterator = await ctx.stub.getStateByRange('', '');
        let result = await iterator.next();
        while (!result.done) {
            const strValue = Buffer.from(result.value.value.toString()).toString('utf8');
            let record;
            try {
                record = JSON.parse(strValue);
            } catch (err) {
                console.log(err);
                record = strValue;
            }
            if (record.id.startsWith("wallet_")) {
                allResults.push(record);
            }
            result = await iterator.next();
        }
        return JSON.stringify(allResults);
    }

    async ProcessTransaction(ctx, id, amount) {
        const walletString = await this.ReadKey(ctx, id);
        let wallet = JSON.parse(walletString);

        const newBalance = wallet.balance + parseFloat(amount);
        if (newBalance < 0.0) {
            throw Error(`Wallet with id: ${id} does not have enough balance.\nBalance: ${wallet.balance}, Amount: ${amount}`);
        }
        wallet = {
            ...wallet,
            balance: newBalance
        };
        await ctx.stub.putState(id, Buffer.from(stringify(sortKeysRecursive(wallet))));
        return JSON.stringify(wallet);
    }

    async ProcessRewards(ctx, rewardsString) {
        const rewards = JSON.parse(rewardsString);
        const wallets = []
        for (const reward of rewards) {
            const walletString = await this.ProcessTransaction(ctx, reward.walletId, reward.reward)
            const wallet = JSON.parse(walletString);
            wallets.push(wallet);
        }
        return JSON.stringify(wallets);
    }

    async UpdatePrice(ctx, newPrice) {
        const priceInfoString = await this.ReadKey(ctx, "priceInfo");
        let priceInfo = JSON.parse(priceInfoString);
        priceInfo.price = parseFloat(newPrice);
        await ctx.stub.putState(priceInfo.id, Buffer.from(stringify(sortKeysRecursive(priceInfo))));
        return JSON.stringify(priceInfo)
    }
}

module.exports = TokenTransfer