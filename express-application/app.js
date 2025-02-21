"use strict";

const {
	TokenApp,
} = require("../token-transfer/token-transfer-application/tokenApp");
const tokenApp = new TokenApp();

const {
	ModelApp,
} = require("../model-propose/model-propose-application/modelApp");
const modelApp = new ModelApp();

const { exec } = require('child_process');

const express = require("express");
const bodyParser = require("body-parser");
const app = express();
app.use(express.json({ limit: "50mb", extended: true }));
const jsonParser = bodyParser.json();
const port = 3000;

const axios = require("axios");

const crypto = require("crypto");
const grpc = require("@grpc/grpc-js");
const {
	connect,
	Contract,
	Identity,
	Signer,
	signers,
} = require("@hyperledger/fabric-gateway");
const fs = require("fs/promises");
const path = require("path");
const { start } = require("repl");

const mspId = "Org1MSP";

const cryptoPath = path.resolve(
	__dirname,
	"..",
	"test-network",
	"organizations",
	"peerOrganizations",
	"org1.example.com"
);
const keyDirPath = path.resolve(
	cryptoPath,
	"users",
	"User1@org1.example.com",
	"msp",
	"keystore"
);
const certPath = path.resolve(
	cryptoPath,
	"users",
	"User1@org1.example.com",
	"msp",
	"signcerts",
	"User1@org1.example.com-cert.pem"
);
const tlsCertPath = path.resolve(
	cryptoPath,
	"peers",
	"peer0.org1.example.com",
	"tls",
	"ca.crt"
);

const peerEndPoint = "localhost:7051";
const peerHostAlias = "peer0.org1.example.com";

const contractToken = InitConnection("main", "tokenCC");
const contractModel = InitConnection("main", "modelCC");

const numNodes = 4;
const rounds = 2;
let currentRound = 0;
const basePrice = 50;
const scale = 10;
const totalRewards = 300;
const aggregatorPort = 8080;

async function newGrpcConnection() {
	const tlsRootCert = await fs.readFile(tlsCertPath);
	const tlsCredentials = grpc.credentials.createSsl(tlsRootCert);
	return new grpc.Client(peerEndPoint, tlsCredentials, {
		"grpc.ssl_target_name_override": peerHostAlias,
		"grpc.max_send_message_length": 100 * 1024 * 1024,
		"grpc.max_receive_message_length": 100 * 1024 * 1024,
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

async function callAggregator() {
	const submits = await modelApp.getAllModels(contractModel);
	const priceInfo = await tokenApp.readKey(contractToken, "priceInfo");
	await axios({
		method: "post",
		url: `http://localhost:${aggregatorPort}/aggregate/`,
		headers: {},
		data: {
			...priceInfo,
			submits: submits,
		},
	});
}

async function processRewards(newPrice, submits) {
	await tokenApp.updatePrice(contractToken, newPrice.toString());
	await tokenApp.processRewards(contractToken, JSON.stringify(submits));
    await modelApp.deleteAllModels(contractModel);
    await startRound(submits);
}

async function startRound(submits) {
	currentRound += 1;
	if (currentRound <= rounds) {
		for (const submit of submits) {
			const port = 8000 + parseInt(submit.walletId[submit.walletId.length - 1]);
			axios({
				method: "post",
				url: `http://localhost:${port}/round/`,
				headers: {},
				data: {
					modelPath: submit["modelPath"],
				},
			});
		}
		console.log(`*** ROUND ${currentRound} STARTED ***`);
		console.log("Training started.");
	} else {
		console.log("All ROUNDS COMPLETED.");
		exec("python3 ../stop.py", (error, stdout, stderr) => {
			if (error) {
				console.error(`Error: ${error.message}`);
				return;
			}
			if (stderr) {
				console.error(`stderr: ${stderr}`);
				return;
			}
			console.log(`stdout: ${stdout}`);
		});
		currentRound = 0;
	}
}

app.get("/", (req, res) => {
	res.send("Hello World! from demo.");
});

app.get("/exit", (req, res) => {
	process.exit();
});

/*
Token App API
 */

app.post("/api/wallets/", async (req, res) => {
	const message = await tokenApp.initWallets(
		contractToken,
		numNodes.toString(),
		basePrice.toString(),
		scale.toString(),
		totalRewards.toString()
	);
	res.send(message);
});

app.post("/api/wallet/", jsonParser, async (req, res) => {
	const message = await tokenApp.createWallet(
		contractToken,
		req.body.id,
		req.body.balance.toString()
	);
	res.send(message);
});

app.get("/api/wallet/", jsonParser, async (req, res) => {
	const message = await tokenApp.readKey(contractToken, req.body.id);
	res.send(message);
});

app.get("/api/wallets/", jsonParser, async (req, res) => {
	const message = await tokenApp.getAllWallets(contractToken);
	res.send(message);
});

app.post("/api/transaction/", jsonParser, async (req, res) => {
	const message = await tokenApp.processTransaction(
		contractToken,
		req.body.id,
		req.body.amount.toString()
	);
	res.send(message);
});

/*
Model App API
*/

app.post("/api/init/", jsonParser, async (req, res) => {
	const message = await modelApp.initRoundInfo(
		contractModel,
		numNodes.toString()
	);
	res.send(message);
});

app.post("/api/model/", jsonParser, async (req, res) => {
	const message = await modelApp.createModel(
		contractModel,
		req.body.id,
		req.body.walletId,
		req.body.path,
		req.body.testDataPath
	);
	if (message) {
		setTimeout(callAggregator, 1);
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
app.post("/api/start/", jsonParser, async (req, res) => {
	currentRound = 1;
	for (let i = 0; i < numNodes; i++)
		axios({
			method: "post",
			url: `http://localhost:${8000 + i}/round/`,
			headers: {},
			data: {
				modelPath: req.body.globalModelPath,
			},
		});
    console.log(`*** ROUND ${currentRound} STARTED ***`);
    console.log("Training started.");
    res.send("Training started.")
});

app.post("/api/aggregator/", jsonParser, async (req, res) => {
	setTimeout(processRewards, 1, req.body.newPrice, req.body.submits);
	res.send("Processing started.");
});

/*
Listen
*/

app.listen(port, () => {
	console.log(`Server is listening on localhost:${port}.\n`);
});
