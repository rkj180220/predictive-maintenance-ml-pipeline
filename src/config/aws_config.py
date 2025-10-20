"""
AWS Configuration for Predictive Maintenance ML Pipeline
Author: ramkumarjayakumar
Date: 2025-10-18
"""

import os
import boto3
from botocore.exceptions import ClientError
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class AWSConfig:
    """AWS Configuration and Client Manager"""

    def __init__(self):
        """Initialize AWS configuration"""
        self.region = os.getenv('AWS_REGION', 'us-east-1')
        self.s3_bucket = os.getenv('AWS_S3_BUCKET', 'predictive-maintenance-data')
        self.cloudwatch_log_group = os.getenv('AWS_CLOUDWATCH_LOG_GROUP', '/aws/predictive-maintenance/pipeline')
        self.cloudwatch_log_stream = os.getenv('AWS_CLOUDWATCH_LOG_STREAM', 'pipeline-execution')

        # AWS Credentials (from environment or AWS credentials file)
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')

        # Initialize clients
        self._s3_client = None
        self._cloudwatch_client = None
        self._cloudwatch_logs_client = None

    @property
    def s3_client(self):
        """Get or create S3 client"""
        if self._s3_client is None:
            try:
                if self.aws_access_key and self.aws_secret_key:
                    self._s3_client = boto3.client(
                        's3',
                        region_name=self.region,
                        aws_access_key_id=self.aws_access_key,
                        aws_secret_access_key=self.aws_secret_key
                    )
                else:
                    # Use default credentials from ~/.aws/credentials
                    self._s3_client = boto3.client('s3', region_name=self.region)
                logger.info("S3 client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize S3 client: {e}")
                self._s3_client = None
        return self._s3_client

    @property
    def cloudwatch_client(self):
        """Get or create CloudWatch client"""
        if self._cloudwatch_client is None:
            try:
                if self.aws_access_key and self.aws_secret_key:
                    self._cloudwatch_client = boto3.client(
                        'cloudwatch',
                        region_name=self.region,
                        aws_access_key_id=self.aws_access_key,
                        aws_secret_access_key=self.aws_secret_key
                    )
                else:
                    self._cloudwatch_client = boto3.client('cloudwatch', region_name=self.region)
                logger.info("CloudWatch client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize CloudWatch client: {e}")
                self._cloudwatch_client = None
        return self._cloudwatch_client

    @property
    def cloudwatch_logs_client(self):
        """Get or create CloudWatch Logs client"""
        if self._cloudwatch_logs_client is None:
            try:
                if self.aws_access_key and self.aws_secret_key:
                    self._cloudwatch_logs_client = boto3.client(
                        'logs',
                        region_name=self.region,
                        aws_access_key_id=self.aws_access_key,
                        aws_secret_access_key=self.aws_secret_key
                    )
                else:
                    self._cloudwatch_logs_client = boto3.client('logs', region_name=self.region)
                logger.info("CloudWatch Logs client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize CloudWatch Logs client: {e}")
                self._cloudwatch_logs_client = None
        return self._cloudwatch_logs_client

    def create_s3_bucket_if_not_exists(self) -> bool:
        """Create S3 bucket if it doesn't exist"""
        if not self.s3_client:
            logger.warning("S3 client not available, skipping bucket creation")
            return False

        try:
            self.s3_client.head_bucket(Bucket=self.s3_bucket)
            logger.info(f"S3 bucket '{self.s3_bucket}' already exists")
            return True
        except ClientError:
            try:
                if self.region == 'us-east-1':
                    self.s3_client.create_bucket(Bucket=self.s3_bucket)
                else:
                    self.s3_client.create_bucket(
                        Bucket=self.s3_bucket,
                        CreateBucketConfiguration={'LocationConstraint': self.region}
                    )
                logger.info(f"S3 bucket '{self.s3_bucket}' created successfully")
                return True
            except ClientError as e:
                logger.error(f"Failed to create S3 bucket: {e}")
                return False

    def create_cloudwatch_log_group_if_not_exists(self) -> bool:
        """Create CloudWatch log group if it doesn't exist"""
        if not self.cloudwatch_logs_client:
            logger.warning("CloudWatch Logs client not available, skipping log group creation")
            return False

        try:
            self.cloudwatch_logs_client.create_log_group(logGroupName=self.cloudwatch_log_group)
            logger.info(f"CloudWatch log group '{self.cloudwatch_log_group}' created successfully")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceAlreadyExistsException':
                logger.info(f"CloudWatch log group '{self.cloudwatch_log_group}' already exists")
                return True
            else:
                logger.error(f"Failed to create CloudWatch log group: {e}")
                return False

    def create_cloudwatch_log_stream_if_not_exists(self) -> bool:
        """Create CloudWatch log stream if it doesn't exist"""
        if not self.cloudwatch_logs_client:
            logger.warning("CloudWatch Logs client not available, skipping log stream creation")
            return False

        try:
            self.cloudwatch_logs_client.create_log_stream(
                logGroupName=self.cloudwatch_log_group,
                logStreamName=self.cloudwatch_log_stream
            )
            logger.info(f"CloudWatch log stream '{self.cloudwatch_log_stream}' created successfully")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceAlreadyExistsException':
                logger.info(f"CloudWatch log stream '{self.cloudwatch_log_stream}' already exists")
                return True
            else:
                logger.error(f"Failed to create CloudWatch log stream: {e}")
                return False

    def setup_aws_resources(self) -> bool:
        """Setup all required AWS resources"""
        results = []
        results.append(self.create_s3_bucket_if_not_exists())
        results.append(self.create_cloudwatch_log_group_if_not_exists())
        results.append(self.create_cloudwatch_log_stream_if_not_exists())
        return all(results)


# Global AWS config instance
aws_config = AWSConfig()

