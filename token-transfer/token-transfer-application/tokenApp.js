'use strict';

const {TextDecoder} = require("util");

class TokenApp {
    constructor() {
        this.utf8decoder = new TextDecoder();
    }

    async initWallets(contract, numNodes, basePrice, scale, totalRewards) {
        try {
            await (await contract).submitTransaction("InitWallets", numNodes, basePrice, scale, totalRewards);
            return "Wallets are successfully initialized.\n";
        } catch (error) {
            console.log(error);
            return error;
        }
    }

    async createWallet(contract, id, balance) {
        try {
            await (await contract).submitTransaction("CreateWallet", id, balance);
            return "Wallet is successfully created.\n";
        } catch (error) {
            console.log(error);
            return error;
        }
    }

    async readKey(contract, id) {
        try {
            const valueBytes = await (await contract).evaluateTransaction("ReadKey", id);
            const valueString = this.utf8decoder.decode(valueBytes);
            return JSON.parse(valueString);
        } catch (error) {
            console.log(error);
            return error;
        }
    }

    async getAllWallets(contract) {
        try {
            const walletsBytes = await (await contract).evaluateTransaction("GetAllWallets");
            const walletsString = this.utf8decoder.decode(walletsBytes);
            return JSON.parse(walletsString);
        } catch (error) {
            console.log(error);
            return error;
        }
    }

    async processTransaction(contract, id, amount) {
        try {
            const walletBytes = await (await contract).submitTransaction("ProcessTransaction", id, amount);
            const walletString = this.utf8decoder.decode(walletBytes);
            return JSON.parse(walletString);
        } catch (error) {
            console.log(error);
            return error;
        }
    }

    async updatePrice(contract, newPrice) {
        try {
            const priceInfoBytes = await (await contract).submitTransaction("UpdatePrice", newPrice);
            const priceInfoString = this.utf8decoder.decode(priceInfoBytes);
            return JSON.parse(priceInfoString);
        } catch (error) {
            console.log(error);
            return error;
        }
    }

    async processRewards(contract, rewards) {
        try {
            const walletsBytes = await (await contract).submitTransaction("ProcessRewards", rewards);
            const walletsString = this.utf8decoder.decode(walletsBytes);
            return JSON.parse(walletsString);
        } catch (error) {
            console.log(error);
            return error;
        }
    }
}

module.exports = {
    TokenApp
}