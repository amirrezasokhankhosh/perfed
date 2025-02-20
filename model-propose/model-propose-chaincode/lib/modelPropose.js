'use strict';

const stringify = require("json-stringify-deterministic");
const sortKeysRecursive = require("sort-keys-recursive");
const { Contract } = require("fabric-contract-api");

class ModelPropose extends Contract {
    async InitRoundInfo(ctx, numNodes) {
        const roundInfo = {
            id: "roundInfo",
            numNodes: parseInt(numNodes),
            remainig: parseInt(numNodes)
        }
        await ctx.stub.putState(roundInfo.id, Buffer.from(stringify(sortKeysRecursive(roundInfo))));
    }

    async ModelExists(ctx, id) {
        const modelBytes = await ctx.stub.getState(id);
        return modelBytes && modelBytes.length > 0;
    }

    async UpdateRoundInfo(ctx) {
        const roundInfoBytes = await ctx.stub.getState("roundInfo");
        const roundInfoString = roundInfoBytes.toString();
        const roundInfo = JSON.parse(roundInfoString);
        roundInfo.remainig = roundInfo.remainig - 1;
        const res = (roundInfo.remainig === 0);
        roundInfo.remainig = res ? roundInfo.numNodes : roundInfo.remainig;
        await ctx.stub.putState("roundInfo", Buffer.from(stringify(sortKeysRecursive(roundInfo))));
        return res;
    }

    async CreateModel(ctx, id, walletId, path, testDataPath) {
        const exists = await this.ModelExists(ctx, id);
        if (exists) {
            throw Error(`A model already exists with id: ${id}`);
        }

        const model = {
            id: id,
            walletId: walletId,
            path: path,
            testDataPath: testDataPath,
        }

        await ctx.stub.putState(model.id, Buffer.from(stringify(sortKeysRecursive(model))));
        return await this.UpdateRoundInfo(ctx);
    }

    async GetAllModels(ctx) {
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
            if (record.id.startsWith("model_")) {
                allResults.push(record);
            }
            result = await iterator.next();
        }
        return JSON.stringify(allResults);
    }

    async ReadModel(ctx, id) {
        const modelBytes = await ctx.stub.getState(id);
        if (! modelBytes || modelBytes.length === 0) {
            throw Error(`No model exists with id ${id}`);
        }
        return modelBytes.toString()
    }
    
    async DeleteAllModels(ctx) {
        const roundInfoBytes = await ctx.stub.getState("roundInfo");
        const roundInfoString = roundInfoBytes.toString()
        const roundInfo = JSON.parse(roundInfoString);
        
        for (let i = 0 ; i < roundInfo.numNodes ; i++) {
            await ctx.stub.deleteState(`model_${i}`);
        }
    }
}

module.exports = ModelPropose