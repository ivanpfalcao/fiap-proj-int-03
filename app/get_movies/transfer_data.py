import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

# Load variables from environment
local_folder = os.getenv('LOCAL_FOLDER')
bucket_name = os.getenv('MINIO_BUCKET')
bucket_folder = os.getenv('MINIO_BUCKET_FOLDER')
minio_endpoint = os.getenv('MINIO_ENDPOINT')
minio_access_key = os.getenv('MINIO_ACCESS_KEY')
minio_secret_key = os.getenv('MINIO_SECRET_KEY')
minio_use_ssl = os.getenv('MINIO_USE_SSL', 'true').lower() == 'true'


s3_client = boto3.client(
	's3',
	endpoint_url=minio_endpoint,
	aws_access_key_id=minio_access_key,
	aws_secret_access_key=minio_secret_key,
	use_ssl=minio_use_ssl
)


def upload_files_to_minio(local_folder, bucket_name, bucket_folder):
	for root, dirs, files in os.walk(local_folder):
		for file in files:
			file_path = os.path.join(root, file)
			object_name = os.path.join(bucket_folder, file)

			try:
				# Upload the file to the bucket
				s3_client.upload_file(file_path, bucket_name, object_name)
				print(f"Successfully uploaded {file_path} to {bucket_name}/{object_name}")
			except (NoCredentialsError, ClientError) as exc:
				print(f"Error occurred while uploading {file_path}: {exc}")

# Run the upload
upload_files_to_minio(local_folder, bucket_name, bucket_folder)