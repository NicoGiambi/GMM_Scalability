# Gaussian Mixture Model
This project propose a study on the scalability of the Gaussian Mixture Model, an unsupervised machine learning technique.
It's a probability-based clustering technique that we explored in different versions: sequential, parallel and distributed.
In particular, in the code we report our distributed version, Mllib GMM version, SGD GMM.

# Preprocessing
In python-scripts folder you can find Python code to extract and augment COCO dataset using pycocotools.

# Tools
All the versions are written in Scala 2.12.10, while for distributed versions we use Spark 3.1.3.
Cloud computing is performed with Google Cloud Platform, using Google Cloud Storage and Google Dataproc.
The gcp folder contains the bash scripts to allocate resources and run scripts on Google Cloud Platform.

# Credits
Giambi Nico, Gaspari Michele
