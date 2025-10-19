# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tempfile
from urllib import parse
from numpy.lib.npyio import save
import ray
import os
import pickle 
import urllib
from urllib.parse import urlparse

import warnings
from typing import Union
from enum import Enum
import torch
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType

import io
import os 

class Cloud(Enum):
    AWS = "aws"
    GCP = "gcp"

def uploadDirectoryToS3(path,bucketname, prefix):
    import boto3

    s3 = boto3.client("s3")
    for root,dirs,files in os.walk(path):
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), path)
            s3_key = os.path.join(prefix, relative_path)
            s3.upload_file(os.path.join(root,file),bucketname,s3_key)

def upload_file_to_s3(path, bucketname, prefix):
    import boto3 
        
    s3 = boto3.client("s3")
    s3.upload_file(path, bucketname, prefix)
    
def upload_dir_to_anyscale(local_path, remote_key):
    save_bucket, remote_prefix, cloud = _get_anyscale_bucket_and_file_key(remote_key)
    if cloud == Cloud.AWS:
        uploadDirectoryToS3(local_path, save_bucket, remote_prefix)
    elif cloud == Cloud.GCP:
        upload_directory_to_gcs(local_path, save_bucket, remote_prefix)
    else:
        raise NotImplementedError

def upload_file_to_anyscale(local_path, remote_key):
    save_bucket, remote_prefix, cloud = _get_anyscale_bucket_and_file_key(remote_key)
    if cloud == Cloud.AWS:
        upload_file_to_s3(local_path, save_bucket, remote_prefix)
    elif cloud == Cloud.GCP:
        upload_file_to_gcs(local_path, save_bucket, remote_prefix)
    else:
        raise NotImplementedError


def _get_anyscale_bucket_and_file_key(path):
    parsed_url = urlparse(os.environ["ANYSCALE_ARTIFACT_STORAGE"])
    if parsed_url.scheme == "s3": 
        cloud = Cloud.AWS
    else:
        cloud = Cloud.GCP
    save_bucket, prefix = parsed_url.netloc, parsed_url.path
    prefix = prefix.lstrip("/")
    save_bucket = save_bucket.rstrip("/")
    path = os.path.join(prefix, path)
    return save_bucket, path, cloud

def write_to_s3(obj, path: str):
    import boto3 

    save_bucket, path, _ = _get_anyscale_bucket_and_file_key(path)
    s3 = boto3.client("s3")
    cpu_buffer = io.BytesIO()
    torch.save(obj, cpu_buffer) # save to cpu
    cpu_buffer.seek(0)
    s3.upload_fileobj(cpu_buffer, save_bucket, path)
    cpu_buffer.close()


ONEGB = 1024 * 1024 * 1024
# Upload a single file to Google Cloud Storage
def upload_file_to_gcs(local_file_path, bucket_name, destination_blob_path):
    import os
    os.environ["GOOGLE_RESUMABLE_MEDIA_PARALLEL_COMPOSITE_UPLOAD_THRESHOLD"] = str(100 * 1024 * 1024)  # 100 MiB threshold
    os.environ["GOOGLE_RESUMABLE_MEDIA_PARALLEL_COMPOSITE_PARTS"] = "10"  # 10 parts in parallel
    from google.cloud import storage
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_path, chunk_size=ONEGB)
    
    blob.upload_from_filename(local_file_path)
    
    print(f"File {local_file_path} uploaded to gs://{bucket_name}/{destination_blob_path}")

# Upload an entire directory to Google Cloud Storage
def upload_directory_to_gcs(local_directory, bucket_name, destination_prefix=""):
    import os
    os.environ["GOOGLE_RESUMABLE_MEDIA_PARALLEL_COMPOSITE_UPLOAD_THRESHOLD"] = str(100 * 1024 * 1024)  # 100 MiB threshold
    os.environ["GOOGLE_RESUMABLE_MEDIA_PARALLEL_COMPOSITE_PARTS"] = "10"  # 10 parts in parallel
    from google.cloud import storage
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    for root, dirs, files in os.walk(local_directory):
        for file in files:
            local_path = os.path.join(root, file)
            
            # Determine the blob path in GCS
            relative_path = os.path.relpath(local_path, local_directory)
            blob_path = os.path.join(destination_prefix, relative_path).replace("\\", "/")  # Ensure proper path separators
            
            # Upload the file
            blob = bucket.blob(blob_path, chunk_size=ONEGB)
            blob.upload_from_filename(local_path)
            
            print(f"File {local_path} uploaded to gs://{bucket_name}/{blob_path}")
    
    print(f"Directory upload complete")

def upload_to_remote_background(config, global_step, local_global_step_folder, main_rank_latest_checkpointed_iteration):
    import time 
            
    def _upload_to_remote_background(config, global_step, local_global_step_folder):
        dir_path = os.path.join(config.trainer.remote_upload_dir, f"global_step_{global_step}")
        print(f"Uploading checkpoint to path: {dir_path}")
        s = time.time()
        upload_dir_to_anyscale(local_global_step_folder, dir_path)
        e = time.time()
        print(f"took {e - s} to upload") 
    
    # only upload on main rank/ caller
    file_path = os.path.join(config.trainer.remote_upload_dir, 'latest_checkpointed_iteration.txt')
    upload_file_to_anyscale(main_rank_latest_checkpointed_iteration, file_path)

    # use num_cpus > 0 to schedule only on worker nodes
    remote_func = ray.remote(num_cpus=0.01, scheduling_strategy="SPREAD")(_upload_to_remote_background)

    return [remote_func.remote(config, global_step, local_global_step_folder) for _ in range(config.trainer.nnodes)]