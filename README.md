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

# Preprocessing
In python-scripts folder you can find Python code to extract and augment COCO dataset using pycocotools.

# Tools
All the versions are written in Scala 2.12.10, while for distributed versions we use Spark 3.1.3.
To make linea algebra computation we use Breeze 2.0.1
Cloud computing is performed with Google Cloud Platform, using Google Cloud Storage and Google Dataproc.
The gcp folder contains the bash scripts to allocate resources and run scripts on Google Cloud Platform.

# Credits
Giambi Nico, Gaspari Michele
