#!/bin/bash

source ./00.variables.sh

gsutil cp ${DATASET_PATH}/GMM_Scalability-assembly-0.2.jar gs://${GCS_BUCKET_NAME}
gsutil cp ${DATASET_PATH}/dataset_0_scaled.txt gs://${GCS_BUCKET_NAME}
gsutil cp ${DATASET_PATH}/scales_0.txt gs://${GCS_BUCKET_NAME}