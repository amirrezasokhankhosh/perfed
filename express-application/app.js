'use strict';


const {TokenApp} = require("../token-transfer/token-transfer-application/tokenApp");
const tokenApp = new TokenApp();

const {ModelApp} = require("../model-propose/model-propose-application/modelApp");
const modelApp = new ModelApp();

const express = require('express');
const bodyParser = require('body-parser');
const app = express();
app.use(express.json({limit: '50mb', extended: true}));
const jsonParser = bodyParser.json();
const port = 3000;

const axios = require("axios")

const crypto = require("crypto");
const grpc = require("@grpc/grpc-js");
const {connect, Contract, Identity, Signer, signers} = require("@hyperledger/fabric-gateway");
const fs = require("fs/promises");
const path = require("path");

const mspId = "Org1MSP";

const cryptoPath = path.resolve(__dirname, '..', 'test-network', 'organizations', 'peerOrganizations', 'org1.example.com');
const keyDirPath = path.resolve(cryptoPath, 'users', 'User1@org1.example.com', 'msp', 'keystore');
const certPath = path.resolve(cryptoPath, 'users', 'User1@org1.example.com', 'msp', 'signcerts', 'User1@org1.example.com-cert.pem');
const tlsCertPath = path.resolve(cryptoPath, 'peers', 'peer0.org1.example.com', 'tls', 'ca.crt');

const peerEndPoint = "localhost:7051";
const peerHostAlias = "peer0.org1.example.com";

const contractToken = InitConnection("main", "tokenCC");
const contractModel = InitConnection("main", "modelCC");

const numNodes = 4;

async function newGrpcConnection() {
    const tlsRootCert = await fs.readFile(tlsCertPath);
    const tlsCredentials = grpc.credentials.createSsl(tlsRootCert);
    return new grpc.Client(peerEndPoint, tlsCredentials, {
        'grpc.ssl_target_name_override': peerHostAlias,
        'grpc.max_send_message_length' : 100 * 1024 * 1024,
        'grpc.max_receive_message_length' : 100 * 1024 * 1024
    });
}

async function newIdentity() {
    const credentials = await fs.readFile(certPath);
    return { mspId, credentials };
}

async function newSigner() {
    const files = await fs.readdir(keyDirPath);
    const keyPath = path.resolve(keyDirPath, files[0]);
    const privateKeyPem = await fs.readFile(keyPath);
    const privateKey = crypto.createPrivateKey(privateKeyPem);
    return signers.newPrivateKeySigner(privateKey);
}

async function InitConnection(channelName, chaincodeName) {
    /*
    * Returns a contract for a given channel and chaincode.
    * */
    const client = await newGrpcConnection();

    const gateway = connect({
        client,
        identity: await newIdentity(),
        signer: await newSigner(),
        // Default timeouts for different gRPC calls
        evaluateOptions: () => {
            return { deadline: Date.now() + 500000 }; // 5 seconds
        },
        endorseOptions: () => {
            return { deadline: Date.now() + 1500000 }; // 15 seconds
        },
        submitOptions: () => {
            return { deadline: Date.now() + 500000 }; // 5 seconds
        },
        commitStatusOptions: () => {
            return { deadline: Date.now() + 6000000 }; // 1 minute
        },
    });

    const network = gateway.getNetwork(channelName);

    return network.getContract(chaincodeName);
}

app.get('/', (req, res) => {
    res.send("Hello World! from demo.");
});

app.get('/exit', (req, res) => {
    process.exit();
});

/*
Token App API
 */

app.post('/api/tokens/', async (req, res) => {
    const message = await tokenApp.initWallets(contractToken, numNodes.toString());
    res.send(message);
});

app.post('/api/token/', jsonParser, async (req, res) => {
    const message = await tokenApp.createWallet(contractToken, req.body.id, req.body.balance.toString());
    res.send(message);
});

app.get('/api/token/', jsonParser, async (req, res) => {
    const message = await tokenApp.readWallet(contractToken, req.body.id);
    res.send(message);
});

app.get('/api/tokens/', jsonParser, async (req, res) => {
    const message = await tokenApp.getAllWallets(contractToken);
    res.send(message);
});

app.post('/api/transaction/', jsonParser, async (req, res) => {
    const message = await tokenApp.processTransaction(contractToken, req.body.id, req.body.amount.toString());
    res.send(message);
});

/*
Model App API
*/

app.post("/api/init/", jsonParser, async (req, res) => {
    const message = await modelApp.initRoundInfo(contractModel, numNodes.toString());
    res.send(message);
});

app.post("/api/model/", jsonParser, async (req, res) => {
    const message = await modelApp.createModel(contractModel, req.body.id, req.body.walletId, req.body.path, req.body.testDataPath);
    if (message) {
        console.log("All models are submitted.")
    }
    res.send(message);
});

app.get("/api/model/", jsonParser, async (req, res) => {
    const message = await modelApp.readModel(contractModel, req.body.id);
    res.send(message);
});

app.get("/api/models/", async (req, res) => {
    const message = await modelApp.getAllModels(contractModel);
    res.send(message);
});

app.delete("/api/models/", jsonParser, async (req, res) => {
    const message = await modelApp.deleteAllModels(contractModel);
    res.send(message);
});

/*
App Control
*/
app.get("/api/start/", async (req, res) => {
    for (let i = 0 ; i < numNodes ; i++)
        await axios.get(`http://localhost:${8000 + i}/round/`);
    res.send("All nodes have started training.")
})


/*
Listen
*/

app.listen(port, () => {
    console.log(`Server is listening on localhost:${port}.\n`);
});