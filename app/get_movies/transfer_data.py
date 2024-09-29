import os
import logging
import boto3
from botocore.exceptions import NoCredentialsError, ClientError



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load variables from environment
local_folder = os.getenv('LOCAL_FOLDER')
bucket_name = os.getenv('S3_BUCKET')
bucket_folder = os.getenv('S3_BUCKET_FOLDER')
S3_endpoint = os.getenv('S3_ENDPOINT')
S3_access_key = os.getenv('S3_ACCESS_KEY')
S3_secret_key = os.getenv('S3_SECRET_KEY')
S3_use_ssl = os.getenv('S3_USE_SSL', 'true').lower() == 'true'


s3_client = boto3.client(
	's3',
	endpoint_url=S3_endpoint,
	aws_access_key_id=S3_access_key,
	aws_secret_access_key=S3_secret_key,
	use_ssl=S3_use_ssl
)


def upload_files_to_minio(local_folder, bucket_name, bucket_folder):
	for root, dirs, files in os.walk(local_folder):
		for file in files:
			file_path = os.path.join(root, file)
			object_name = os.path.join(bucket_folder, file)

			try:
				# Upload the file to the bucket
				s3_client.upload_file(file_path, bucket_name, object_name)
				logging.info(f"Successfully uploaded {file_path} to {bucket_name}/{object_name}")
			except (NoCredentialsError, ClientError) as exc:
				logging.error(f"Error occurred while uploading {file_path}: {exc}")


upload_files_to_minio(local_folder, bucket_name, bucket_folder)