import ClusteringUtils._
import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.{GaussianMixture, GaussianMixtureModel}
import org.apache.spark.mllib.linalg.{Matrices, Vector, Vectors}
import org.apache.spark.rdd.RDD
import com.github.gradientgmm.GradientGaussianMixture
import org.apache.log4j.Logger
import org.apache.log4j.Level

object Benchmark {

  def fitSave(model: String,  // used algorithm among [seq, par, rdd, sgd, mllib]
              outPath: String,  // model save destination
              sc: SparkContext,
              parsedData: Either[Array[(Double, Double)], RDD[Vector]], // dataset
              kPoints: Array[(Double, Double)], // initial centroids
              clusters: Int,  // number of clusters
              maxIter: Int,  // max number of iteration
              tolerance: Double,  // early stopping metric
              scaleX: (Double, Double),
              scaleY: (Double, Double),
              args: Array[String]): Unit = {

    val t1 = System.nanoTime        // start a Timer to record the training duration

    if (model.equals("mllib")) {
      // We initialize the mllib model the same way we do for our GMM versions to grant fairness
      val mvWeights = for (i <- kPoints.indices) yield 1.0 / kPoints.length    // init the pi_k

      val mvGaussians = for (k_p <- kPoints) yield {
        val initMu = Vectors.dense(Array(k_p._1, k_p._2))  // init the centroids
        val initCov = Matrices.dense(2, 2, DenseMatrix.eye[Double](2).toArray) // init the covariance matrix
        new MultivariateGaussian(initMu, initCov)
      }

      // generate a GMM with the previously defined parameters
      val initModel = new GaussianMixtureModel(weights = mvWeights.toArray, gaussians = mvGaussians)

      println("Starting Centroids: ")
      printMllibCentroids(initModel, scaleX, scaleY)
      // initModel.gaussians.map(_.sigma).foreach(println)

      // initialize the model which starts from initModel and fits to our dataset
      val est = new GaussianMixture().
                    setK(clusters).
                    setMaxIterations(maxIter).
                    setConvergenceTol(tolerance).
                    setInitialModel(initModel).
                    run(parsedData match { case Right(x) => x})

      printMllibCentroids(est, scaleX, scaleY)

      // if (!Files.exists(Paths.get(outPath)))
      //  est.save(sc, outPath)
    }
    else if (model.equals("sgd")) {

      val weights = (for (i <- kPoints.indices) yield 1.0 / kPoints.length).toArray
      val initMu = kPoints.map(p => Vectors.dense(Array(p._1, p._2)))  // init the centroids
      val initCov = (for (i <- kPoints.indices) yield Matrices.dense(2, 2, DenseMatrix.eye[Double](2).toArray)).toArray // init the covariance matrix

      val est = GradientGaussianMixture(weights = weights, means = initMu, covs = initCov)
                .setBatchSize(65536*2)
                .setMaxIter(maxIter*50)
                .step(parsedData match { case Right(x) => x})
                .toSparkGMM

      printMllibCentroids(est, scaleX, scaleY)

      // if (!Files.exists(Paths.get(outPath)))
      //  est.save(sc, outPath)
    }
    else if (model.equals("seq")) {
      SequentialGMM.run(parsedData = parsedData match { case Left(x) => x},
                        kPoints = kPoints,
                        scales = Array(scaleX, scaleY),
                        K = clusters,
                        maxIter = maxIter)
    }
    else if (model.equals("par")) {
      ParallelGMM.run(parsedData = parsedData match { case Left(x) => x},
                      kPoints = kPoints,
                      scales = Array(scaleX, scaleY),
                      K = clusters,
                      maxIter = maxIter)
    }
    else if (model.equals("rdd")) {
      DistributedGMM.run(sc = sc,
                         parsedData = parsedData match { case Right(x) => x},
                         kPoints = kPoints,
                         scales = Array(scaleX, scaleY),
                         K = clusters,
                         maxIter = maxIter)
    }

    val duration1 = (System.nanoTime - t1) / 1e9d
    println(model + " duration: " + duration1)

  }

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    val conf = new SparkConf().setAppName("Benchmark").setMaster(args(0))
//    val conf = new SparkConf().setAppName("Benchmark")

    val sc = new SparkContext(conf)

    val scSettings = args(0)
    val model = args(1)
    val clusters = args(2).toInt
    val augmentationFactor = args(3)
    val datasetFolder = args(4)
    val randomSeed = args(5).toInt


    val availableModels = Set("mllib", "sgd", "seq", "par", "rdd")
    assert(availableModels.contains(model), message = "Choose another algorithm among " + availableModels.toString())

    val outPath = "model/" + model + "/"

    val filename = datasetFolder + "dataset_" + augmentationFactor + "_scaled.txt"
    val scalesFilename = datasetFolder + "scales_" + augmentationFactor + ".txt"

    val (maxIter, tolerance) = getHyperparameters

    val scales = sc.textFile(scalesFilename).map(s => s.trim.split(' ').map(_.toDouble)).map(p => (p(0), p(1))).collect()

    // load an initial version of the dataset into memory
    val data = sc.textFile(filename).map(s => Vectors.dense(s.trim.split(' ').map(_.toDouble))).cache()

    // We need RDD[Vector] for [mllib, sgd, rdd] and Array[(Double,Double)] for [seq, par],
    // so to avoid loading a dataset in memory twice we use the Either construct
    val parsedData : Either[Array[(Double, Double)], RDD[Vector]] =
    {
      if (Set("seq", "par").contains(args(1)))
        Left(data.collect().map(p => (p(0), p(1))))
      else
        Right(data)
    }

    // the dataset is scaled in [0,1], we also saved the scales to see the real centroids
    val scaleX = scales(0)
    val scaleY = scales(1)

    // Initial centroids
    val kPoints = data.takeSample(withReplacement = false, clusters, randomSeed).map(p => (p(0), p(1)))

    println("Fitting with " + scSettings + " on " + model + " model with " + clusters + " clusters and augmentation set to " + augmentationFactor)
    fitSave(model=model,
            outPath=outPath,
            sc=sc,
            clusters=clusters,
            maxIter=maxIter,
            parsedData=parsedData,
            kPoints=kPoints,
            tolerance=tolerance,
            scaleX=scaleX,
            scaleY=scaleY,
            args=args)

    println()
    println("------------------------------------")
    println()

//    if (!availableModels.contains(model))
//      println("No available models with this name")
//    else if (model == "KMeans") {
//      val estimator = KMeansModel.load(sc, outPath)
//      val preds = estimator.predict(parsedData)
//      if (!Files.exists(Paths.get(outPath + "/predictions")))
//        preds.saveAsTextFile(outPath + "/predictions")
//      estimator.clusterCenters.sortWith(_ (0) < _ (0)).foreach(println)
//    }
//    else {
//      val estimator = GaussianMixtureModel.load(sc, outPath)
//      val preds = estimator.predict(parsedData)
//      if (!Files.exists(Paths.get(outPath + "/predictions")))
//        preds.saveAsTextFile(outPath + "/predictions")
//      estimator
//        .gaussians
//        .map(_.mu)
//        .map(p => ((p(0) * (scaleX._2 - scaleX._1)) + scaleX._1, (p(1) * (scaleY._2 - scaleY._1)) + scaleY._1))
//        .sortWith(_._1 < _._1)
//        .foreach(println)
//    }

    sc.stop()
  }
}