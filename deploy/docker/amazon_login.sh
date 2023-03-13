set -x

AWS_ACCESS_KEY_ID=$1
AWS_SECRET_ACCESS_KEY=$2
AWS_REGION=$3
AWS_URL=$4
REPO=$5
LANGUAGE=$6

echo "Loggin into ${AWS_URL} at ${AWS_REGION}"
echo "127.0.0.1 docker" >> /etc/hosts
apk add --no-cache aws-cli
aws configure set aws_access_key_id ${AWS_ACCESS_KEY_ID}
aws configure set aws_secret_access_key ${AWS_SECRET_ACCESS_KEY}
aws configure set region "${AWS_REGION}"
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${AWS_URL}"