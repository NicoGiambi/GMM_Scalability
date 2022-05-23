# Gaussian Mixture Model
This project propose a study on the scalability of the Gaussian Mixture Model, an unsupervised machine learning technique.
It's a probability-based clustering technique that we applied on COCO dataset and explored in different versions: 

- sequential
- parallel
- distributed.

In particular, we conduct a benchmark not only among sequential, parallel and distributed versions, but also among
our implementation of GMM with:

- Mllib implementation from https://spark.apache.org/docs/latest/api/scala/org/apache/spark/mllib/clustering/GaussianMixture.html
- SGD GMM implementation from https://www.nestorsag.com/blog/scaling-gaussian-mixture-models-to-massive-datasets/ 

# Tools
All the versions are written in Scala 2.12.10, while for distributed versions we use Spark 3.1.3.
To make linea algebra computations we use Breeze 2.0.1.
Cloud computing is performed with Google Cloud Platform, using Google Cloud Storage and Google Dataproc.

# Project Structure
Source code in Scala+Spark is in src/main/scala folder.
Datasets folder contain the original version of COCO dataset.
Python preprocessing of COCO dataset is python-scripts.
In gcp folder there are bash scripts to allocate resources on Google Cloud Storage and run jobs with Google Dataproc.

# Credits
Giambi Nico, Gaspari Michele
