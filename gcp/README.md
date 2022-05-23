# Google Cloud Platform
In order to run distributed versions on the cloud, launch the scripts in sequence.
Decide the hyperparameters by setting environment variables in 00.variables.sh.
Pay attention to point 3, you can decide whether to run a single-node cluster (3b.dataproc-create-single-node.sh)
or to run a multi-worker cluster (first run 3a.dataproc-create-cluster.sh and then 05.dataproc-scaleup.sh to
get a cluster with DATAPROC_NUMWORKER nodes).
Always remember to deallocate resources!
