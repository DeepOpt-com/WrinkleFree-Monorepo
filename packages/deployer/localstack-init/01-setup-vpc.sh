#!/bin/bash
# LocalStack initialization script
# Creates test VPC, subnet, and security group for integration tests

set -e

echo "Setting up LocalStack test environment..."

# Create VPC
VPC_ID=$(awslocal ec2 create-vpc \
    --cidr-block 10.0.0.0/16 \
    --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=wrinklefree-test},{Key=project,Value=wrinklefree},{Key=environment,Value=test}]' \
    --query 'Vpc.VpcId' \
    --output text)

echo "Created VPC: $VPC_ID"

# Create subnet
SUBNET_ID=$(awslocal ec2 create-subnet \
    --vpc-id "$VPC_ID" \
    --cidr-block 10.0.1.0/24 \
    --availability-zone us-east-1a \
    --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=wrinklefree-test-subnet}]' \
    --query 'Subnet.SubnetId' \
    --output text)

echo "Created Subnet: $SUBNET_ID"

# Create security group
SG_ID=$(awslocal ec2 create-security-group \
    --group-name wrinklefree-test-sg \
    --description "Test security group for WrinkleFree" \
    --vpc-id "$VPC_ID" \
    --query 'GroupId' \
    --output text)

echo "Created Security Group: $SG_ID"

# Add SSH ingress rule (restricted to test network)
awslocal ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp \
    --port 22 \
    --cidr 10.0.0.0/8

# Add inference port ingress rule
awslocal ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp \
    --port 8080 \
    --cidr 0.0.0.0/0

echo "Added security group rules"

# Create internet gateway
IGW_ID=$(awslocal ec2 create-internet-gateway \
    --query 'InternetGateway.InternetGatewayId' \
    --output text)

awslocal ec2 attach-internet-gateway \
    --internet-gateway-id "$IGW_ID" \
    --vpc-id "$VPC_ID"

echo "Created and attached Internet Gateway: $IGW_ID"

# Create S3 bucket for test artifacts
awslocal s3 mb s3://wrinklefree-test-artifacts

echo "Created S3 bucket: wrinklefree-test-artifacts"

# Store configuration for tests
cat > /tmp/localstack-config.json << EOF
{
    "vpc_id": "$VPC_ID",
    "subnet_id": "$SUBNET_ID",
    "security_group_id": "$SG_ID",
    "internet_gateway_id": "$IGW_ID",
    "region": "us-east-1"
}
EOF

echo "LocalStack initialization complete!"
echo "Configuration saved to /tmp/localstack-config.json"
