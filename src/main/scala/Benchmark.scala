import ClusteringUtils._
import breeze.linalg.DenseMatrix
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.{GaussianMixture, GaussianMixtureModel, KMeans}
import org.apache.spark.mllib.linalg.{Matrices, Vector, Vectors}
import org.apache.spark.rdd.RDD
import com.github.gradientgmm.GradientGaussianMixture
import org.apache.hadoop.conf.Configuration

import java.nio.file.{Files, Paths}
import org.apache.log4j.Logger
import org.apache.log4j.Level


object Benchmark {

  def fitSave(model: String,
              outPath: String,
              sc: SparkContext,
              parsedData: RDD[Vector],
              kPoints: Array[(Double, Double)],
              clusters: Int,
              maxIter: Int,
              tolerance: Double,
              scaleX: (Double, Double),
              scaleY: (Double, Double),
              args: Array[String]): Int = {

    val t1 = System.nanoTime

    if (model.equals("KMeans")) {
      val est = new KMeans().setK(clusters).setMaxIterations(maxIter).setEpsilon(tolerance).run(parsedData)
      est.clusterCenters.sortWith(_ (0) < _ (0)).foreach(println)
      //      if (!Files.exists(Paths.get(outPath)))
      //        est.save(sc, outPath)
    }
    else if (model.equals("GMM")) {
      val mvWeights = for (i <- kPoints.indices) yield 1.0 / kPoints.length
      val mvGaussians = for (k_p <- kPoints) yield {
        val initMu = Vectors.dense(Array(k_p._1, k_p._2))
        val initCov = Matrices.dense(2, 2, DenseMatrix.eye[Double](2).toArray)
        new MultivariateGaussian(initMu, initCov)
      }
      val initModel = new GaussianMixtureModel(weights = mvWeights.toArray, gaussians = mvGaussians)
      println("Starting Centroids: ")
      initModel
        .gaussians
        .map(_.mu)
        .map(p => ((p(0) * (scaleX._2 - scaleX._1)) + scaleX._1, (p(1) * (scaleY._2 - scaleY._1)) + scaleY._1))
        .sortWith(_._1 < _._1)
        .foreach(println)
      // initModel.gaussians.map(_.sigma).foreach(println)
      val est = new GaussianMixture().setK(clusters).setMaxIterations(maxIter).setConvergenceTol(tolerance).run(parsedData)
      est
        .gaussians
        .map(_.mu)
        .map(p => ((p(0) * (scaleX._2 - scaleX._1)) + scaleX._1, (p(1) * (scaleY._2 - scaleY._1)) + scaleY._1))
        .sortWith(_._1 < _._1)
        .foreach(println)
      // if (!Files.exists(Paths.get(outPath)))
      //  est.save(sc, outPath)
    }
    else if (model.equals("SGDGMM")) {
      val est = GradientGaussianMixture.fit(data = parsedData, k = clusters, kMeansIters = 0, kMeansTries = 0, maxIter = maxIter).toSparkGMM
      est
        .gaussians
        .map(_.mu)
        .map(p => ((p(0) * (scaleX._2 - scaleX._1)) + scaleX._1, (p(1) * (scaleY._2 - scaleY._1)) + scaleY._1))
        .sortWith(_._1 < _._1)
        .foreach(println)
      // if (!Files.exists(Paths.get(outPath)))
      //  est.save(sc, outPath)
    }
    else if (model.equals("seqGMM")) {
      SequentialGMM.run(args = args)
    }
    else if (model.equals("parGMM")) {
      ParallelGMM.run(args = args)
    }
    else if (model.equals("rddGMM")) {
      DistributedGMM.run(args = args, sc = sc, parsedData = parsedData, Array(scaleX, scaleY))
    }

    val duration1 = (System.nanoTime - t1) / 1e9d
    println(model + " duration: " + duration1)

    0
  }

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    val conf = new SparkConf().setMaster(args(0)).setAppName("Benchmark")
    conf.set("spark.testing.memory", "4294960000")
    val sc = new SparkContext(conf)

    val hadoopConfig: Configuration = sc.hadoopConfiguration
    hadoopConfig.set("fs.hdfs.impl", classOf[org.apache.hadoop.hdfs.DistributedFileSystem].getName)
    hadoopConfig.set("fs.file.impl", classOf[org.apache.hadoop.fs.LocalFileSystem].getName)

    val model = args(1)
    val clusters = args(2).toInt

    val availableModels = Set("KMeans", "GMM", "SGDGMM", "seqGMM", "parGMM", "rddGMM")
    assert(availableModels.contains(args(1)))

    val outPath = "model/" + model + "/"

    val filename = "../../../datasets/dataset_" + args(3) + "_scaled.txt"
    val scalesFilename = "../../../datasets/scales_" + args(3) + ".txt"

//    val filename = "datasets/dataset_" + args(3) + "_scaled.txt"
//    val scalesFilename = "datasets/scales_" + args(3) + ".txt"

    val (maxIter, tolerance, seed) = getHyperparameters()

    val scales = import_files(scalesFilename)
    val data = sc.textFile(filename)
    val parsedData = data.map(s => Vectors.dense(s.trim.split(' ').map(_.toDouble))).cache()
    val scaleX = scales(0)
    val scaleY = scales(1)

    val kPoints = parsedData.takeSample(false, clusters, 42).map(p => (p(0), p(1)))

    println("Fitting with " + args(0) + " on " + args(1) + " model with " + args(2) + " clusters and augmentation set to " + args(3))
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


//----------------------------------------
//
//Dataset 16 -- 12 core
//
//KMeans duration: 93.6907123
//
//[26.12619531736289,29.35331810814975]
//[89.65028233580287,104.61239756541949]
//[153.33654275703222,267.7527081611716]
//[266.9311804979041,145.04751738109738]
//[438.7998353724877,338.2476937506053]
//
//----------------------------------------
//
//Dataset 16 -- 1 core
//
//KMeans duration: 311.3623638
//
//[26.12619531736289,29.35331810814975]
//[89.65028233580287,104.61239756541949]
//[153.33654275703222,267.7527081611716]
//[266.9311804979041,145.04751738109738]
//[438.7998353724877,338.2476937506053]

//----------------------------------------
//
//Dataset 1 -- 1 core
//
//KMeans duration: 13.9155618
//
//[26.12619531736289,29.35331810814975]
//[89.65028233580287,104.61239756541949]
//[153.33654275703222,267.7527081611716]
//[266.9311804979041,145.04751738109738]
//[438.7998353724877,338.2476937506053]

//----------------------------------------
//
//Dataset 1 -- 1 core
//
//GMM duration: 192.7647633
//
//[10.325164928633521,14.085490970190543]
//[25.762871742157678,31.37923392165051]
//[56.38092112134998,65.6480482647342]
//[118.55649811498574,135.31646379648762]
//[279.79615734057495,243.11441659012084]