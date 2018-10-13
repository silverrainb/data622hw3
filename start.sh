#!/usr/bin/env bash

cd /usr/src/app/data622hw3
git pull
cd titanic
python3 main.py

#docker pull silverrainb/titanic
#docker run -it --rm --name trading-app --network mongo-network silverrainb/crypto-forecast
