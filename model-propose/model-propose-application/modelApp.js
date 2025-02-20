'use strict';

const {TextDecoder} = require("util");

class ModelApp {
    constructor () {
        this.utf8decoder = new TextDecoder();
    }

    async initRoundInfo(contract, numNodes) {
        try {
            await (await contract).submitTransaction("InitRoundInfo", numNodes);
            return "RoundInfo has been successfully initialized."
        } catch (error) {
            console.log(error);
            return error;
        }
    }

    async createModel(contract, id, walletId, path, testDataPath, tries=5) {
        while (tries != 0) {
            try {
                const resBytes = await (await contract).submitTransaction("CreateModel", id, walletId, path, testDataPath);
                const resString = this.utf8decoder.decode(resBytes);
                return JSON.parse(resString);
            } catch(error) {
                tries = tries - 1;
                await new Promise(resolve => setTimeout(resolve, 100));
            }
        }
    }

    async getAllModels(contract) {
        try {
            const modelsBytes = await (await contract).evaluateTransaction("GetAllModels");
            const modelsString = this.utf8decoder.decode(modelsBytes)
            return JSON.parse(modelsString);
        } catch (error) {
            console.log(error);
            return error;
        }
    }

    async readModel(contract, id) {
        try {
            const modelBytes = await (await contract).evaluateTransaction("ReadModel", id);
            const modelString = this.utf8decoder.decode(modelBytes)
            return JSON.parse(modelString);
        } catch (error) {
            console.log(error);
            return error;
        }
    }

    async deleteAllModels(contract) {
        try {
            await (await contract).submitTransaction("DeleteAllModels");
            return "All models have been successfully deleted."
        } catch (error) {
            console.log(error);
            return error;
        }
    }
}

module.exports = {
    ModelApp
}