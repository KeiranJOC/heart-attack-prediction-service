#!/bin/bash

# build and containerize model
bentoml build
bentoml containerize heart_attack_prediction_model:uwer7ydadw5dqjq5 --platform=linux/amd64

# push model container to AWS Elastic Container Registry
aws ecr get-login-password --region ap-southeast-2 | docker login --username AWS --password-stdin 843032675284.dkr.ecr.ap-southeast-2.amazonaws.com
docker tag heart_attack_prediction_model:uwer7ydadw5dqjq5 843032675284.dkr.ecr.ap-southeast-2.amazonaws.com/heart-attack-prediction-service:latest
docker push 843032675284.dkr.ecr.ap-southeast-2.amazonaws.com/heart-attack-prediction-service:latest

