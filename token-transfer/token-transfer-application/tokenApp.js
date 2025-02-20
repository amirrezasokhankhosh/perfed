'use strict';

const {TextDecoder} = require("util");

class TokenApp {
    constructor() {
        this.utf8decoder = new TextDecoder();
    }

    async initWallets(contract, numNodes) {
        try {
            await (await contract).submitTransaction("InitWallets", numNodes);
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

    async readWallet(contract, id) {
        try {
            const walletBytes = await (await contract).evaluateTransaction("ReadWallet", id);
            const walletString = this.utf8decoder.decode(walletBytes);
            return JSON.parse(walletString);
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
}

module.exports = {
    TokenApp
}