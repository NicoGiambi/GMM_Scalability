#!/bin/bash

source ./00.variables.sh

./90.log-time.sh "SUBMITTING JOB '${DATAPROC_CLUSTER_NAME}' ..."

gcloud dataproc jobs submit spark \
--cluster=${DATAPROC_CLUSTER_NAME} \
--class=Benchmark \
--jars=gs://${GCS_BUCKET_NAME}/GMM_Scalability-assembly-0.2.jar \
--region=${DATAPROC_CLUSTER_REGION} \
-- ${DATAPROC_NUMWORKERS} ${ALGORITHM} ${GMM_NUMBER_OF_CLUSTERS} ${DATASET} gs://${GCS_BUCKET_NAME}/ ${SEED}
#--scopes storage-rw

./90.log-time.sh "JOB '${DATAPROC_CLUSTER_NAME}' ENDED!"