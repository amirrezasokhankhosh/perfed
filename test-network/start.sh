./network.sh down
./network.sh up
./network.sh createChannel -c main
./network.sh deployCC -ccp ../token-transfer/token-transfer-chaincode -ccn tokenCC -c main -ccl javascript
./network.sh deployCC -ccp ../model-propose/model-propose-chaincode -ccn modelCC -c main -ccl javascript

