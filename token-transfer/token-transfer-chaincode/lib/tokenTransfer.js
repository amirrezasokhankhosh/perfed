'use strict';

const stringify = require("json-stringify-deterministic");
const sortKeysRecursive = require("sort-keys-recursive");
const {Contract} = require("fabric-contract-api");

class TokenTransfer extends Contract {
    async InitWallets(ctx, numNodes) {
        for (let i = 0; i < parseInt(numNodes); i++) {
            const wallet = {
                id: `wallet_${i}`,
                balance: 0.0
            }
            await ctx.stub.putState(wallet.id, Buffer(stringify(sortKeysRecursive(wallet))));
        }
    }

    async WalletExists(ctx, id) {
        const walletBytes = await ctx.stub.getState(id);
        return walletBytes && walletBytes.length > 0;
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

        await ctx.stub.putState(wallet.id, Buffer(stringify(sortKeysRecursive(wallet))));
    }

    async ReadWallet(ctx, id) {
        const walletBytes = await ctx.stub.getState(id);
        if (!walletBytes || walletBytes.length === 0) {
            throw Error(`No wallet exists with id: ${id}.`);
        }
        return walletBytes.toString()
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
        const walletString = await this.ReadWallet(ctx, id);
        let wallet = JSON.parse(walletString);

        const newBalance = wallet.balance + parseFloat(amount);
        if (newBalance < 0.0) {
            throw Error(`Wallet with id: ${id} does not have enough balance.\nBalance: ${wallet.balance}, Amount: ${amount}`);
        }
        wallet = {
            ...wallet,
            balance: newBalance
        };
        await ctx.stub.putState(id, Buffer(stringify(sortKeysRecursive(wallet))));
        return JSON.stringify(wallet);
    }
}

module.exports = TokenTransfer