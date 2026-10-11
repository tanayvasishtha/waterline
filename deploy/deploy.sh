#!/usr/bin/env bash
# Waterline AWS deploy. Runs in AWS CloudShell (docker + aws cli preinstalled).
# Usage: bash deploy/deploy.sh
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
ECR="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/waterline"
TABLE=waterline
BUCKET="waterline-evidence-$ACCOUNT"

echo "== ECR repo =="
aws ecr describe-repositories --repository-names waterline --region "$REGION" 2>/dev/null \
  || aws ecr create-repository --repository-name waterline --region "$REGION" >/dev/null

echo "== Build + push image =="
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR"
docker build -t waterline -f deploy/Dockerfile .
docker tag waterline:latest "$ECR:latest"
docker push "$ECR:latest"

echo "== DynamoDB table =="
aws dynamodb describe-table --table-name "$TABLE" --region "$REGION" 2>/dev/null \
  || aws dynamodb create-table --table-name "$TABLE" --region "$REGION" \
       --attribute-definitions AttributeName=pk,AttributeType=S AttributeName=sk,AttributeType=S \
       --key-schema AttributeName=pk,KeyType=HASH AttributeName=sk,KeyType=RANGE \
       --billing-mode PAY_PER_REQUEST >/dev/null

echo "== Evidence bucket =="
aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" 2>/dev/null || true

echo "== Lambda function =="
ROLE=waterline-lambda-role
aws iam get-role --role-name "$ROLE" >/dev/null 2>&1 || {
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document '{
    "Version":"2012-10-17","Statement":[{"Effect":"Allow",
    "Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "$ROLE" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  aws iam put-role-policy --role-name "$ROLE" --policy-name waterline --policy-document '{
    "Version":"2012-10-17","Statement":[
      {"Effect":"Allow","Action":["dynamodb:PutItem","dynamodb:Query"],"Resource":"*"},
      {"Effect":"Allow","Action":"s3:PutObject","Resource":"*"},
      {"Effect":"Allow","Action":"bedrock:InvokeModel","Resource":"*"}]}'
  sleep 10
}
ROLE_ARN=$(aws iam get-role --role-name "$ROLE" --query Role.Arn --output text)
aws lambda get-function --function-name waterline-poller --region "$REGION" >/dev/null 2>&1 \
  && aws lambda update-function-code --function-name waterline-poller --image-uri "$ECR:latest" --region "$REGION" >/dev/null \
  || aws lambda create-function --function-name waterline-poller --region "$REGION" \
       --package-type Image --code ImageUri="$ECR:latest" --role "$ROLE_ARN" \
       --timeout 300 --memory-size 1024 >/dev/null

echo "== EventBridge schedule (every 15 min) =="
RULE=waterline-poll
aws events put-rule --name "$RULE" --schedule-expression "rate(15 minutes)" --region "$REGION" >/dev/null
aws events put-targets --rule "$RULE" --region "$REGION" --targets \
  "Id"="1","Arn"="arn:aws:lambda:$REGION:$ACCOUNT:function:waterline-poller","Input"="{\"sites\":[\"hancock\",\"in_overland\"]}" >/dev/null
aws lambda add-permission --function-name waterline-poller --region "$REGION" \
  --statement-id waterline-schedule --action lambda:InvokeFunction \
  --principal events.amazonaws.com --source-arn "arn:aws:events:$REGION:$ACCOUNT:rule/$RULE" 2>/dev/null || true

echo "Done. Test with: aws lambda invoke --function-name waterline-poller out.json --region $REGION"
