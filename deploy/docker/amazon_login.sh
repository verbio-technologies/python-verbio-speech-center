set -x

AWS_ACCESS_KEY_ID=$1
AWS_SECRET_ACCESS_KEY=$2
AWS_REGION=$3
AWS_URL=$4
AWS_ECR_ID=$(echo $AWS_URL | cut -f 1 -d '.')
REPO=$5
LANGUAGE=$6

echo "Loggin into ${AWS_URL} at ${AWS_REGION}"
echo "127.0.0.1 docker" >> /etc/hosts
apk add --no-cache aws-cli
aws configure set aws_access_key_id ${AWS_ACCESS_KEY_ID}
aws configure set aws_secret_access_key ${AWS_SECRET_ACCESS_KEY}
aws configure set region "${AWS_REGION}"
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${AWS_URL}"


create_ecr_repository () {
    repository=$1
    url=$2
    ecr_id=$3
    
	echo "Creating repository ${repository} in ${url}"
    export $(printf "AWS_ACCESS_KEY_ID=%s AWS_SECRET_ACCESS_KEY=%s AWS_SESSION_TOKEN=%s" \
        $(aws sts assume-role \
        --role-arn arn:aws:iam::321880733545:role/ECS-AssumeRole-Squad2 \
        --role-session-name AWSCLI-Session-Asr4 \ 
        --query "Credentials.[AccessKeyId,SecretAccessKey,SessionToken]" \
        --output text))
	aws ecr create-repository --registry-id ${ecr_id} --repository-name "${repository}"
}


repository="${REPO}-${LANGUAGE}"
output=$(aws ecr describe-repositories --registry-id ${AWS_ECR_ID} --repository-names "${repository}" 2>&1)
if [ $? -ne 0 ]; then
    if echo ${output} | grep -q RepositoryNotFoundException; then
    create_ecr_repository ${repository} ${AWS_URL} ${AWS_ECR_ID}
    else
	>&2 echo ${output}
    fi
fi


repository="${REPO}-gpu-${LANGUAGE}"
output=$(aws ecr describe-repositories --registry-id ${AWS_ECR_ID} --repository-names "${repository}" 2>&1)
if [ $? -ne 0 ]; then
    if echo ${output} | grep -q RepositoryNotFoundException; then
	create_ecr_repository ${repository} ${AWS_URL} ${AWS_ECR_ID}
    else
	>&2 echo ${output}
    fi
fi
