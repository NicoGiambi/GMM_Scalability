import ClusteringUtils._
import breeze.linalg.{DenseMatrix, DenseVector, _}
import breeze.numerics.log
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD

import scala.annotation.tailrec
import org.apache.spark.{SparkConf, SparkContext}

import java.lang.management.GarbageCollectorMXBean


object DistributedGMM {

  def expMaxStep (sc: SparkContext, points: DenseMatrix[Double], clusters : RDD[Cluster], K : Int):
  (RDD[Cluster], Double) = {

    // Expectation step
    // distribute on columns, i.e. on clusters

    val startTime5 = System.nanoTime

    val gaussians = clusters.map(a => a.gaussian(points, isParallel = false)).collect()

    val duration5 = (System.nanoTime - startTime5) / 1e9d
    println("gaussians: " + duration5)

    val startTime0 = System.nanoTime

    val gamma_nk = new DenseMatrix(points.cols, K, gaussians.flatten)
    val totals = sum(gamma_nk(*, ::))

    val duration0 = (System.nanoTime - startTime0) / 1e9d
    println("gamma_nk: " + duration0)

    val startTime2 = System.nanoTime

    val gaussians_norm = gaussians.flatMap(diag => (DenseVector(diag) / totals).toArray)
    val gamma_nk_norm = new DenseMatrix(points.cols, K, gaussians_norm).t

    val duration2 = (System.nanoTime - startTime2) / 1e9d
    println("normalization: " + duration2)



    val startTime1 = System.nanoTime

    // Maximization step
    val newClusters = clusters.map(_.maximizationStep(points, gamma_nk_norm))

    val duration1 = (System.nanoTime - startTime1) / 1e9d
    println("maximization_step: " + duration1)

    val sampleLikelihood = log(totals)
    val likelihood = sum(sampleLikelihood)

    (newClusters, likelihood)
  }


  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("DistributedGMM").setMaster(args(0))
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.executor.memory", "4g")
    val sc = new SparkContext(conf)

    // number of clusters
    val K = args(2).toInt

    // file with anchors sizes
    val filename = "datasets/dataset_" + args(3) + "_scaled.txt"
    val scalesFilename = "datasets/scales_" + args(3) + ".txt"

    val (maxIter, tolerance, seed) = getHyperparameters()

//    val points = import_files(filename)
//    val scales = import_files(scalesFilename)
//    val data = sc.parallelize(points)
//    val parsedData = data.map(s => Vectors.dense(Array(s._1, s._2))).cache()

    val scales = import_files(scalesFilename)
    val data = sc.textFile(filename)
    val parsedData = data.map(s => Vectors.dense(s.trim.split(' ').map(_.toDouble))).cache()
//    val points = parsedData.collect().map(p => (p(0), p(1)))

    val scaleX = scales(0)
    val scaleY = scales(1)

    println("Number of points:")
//    println(points.length)
    println(parsedData.count())

    // take 5 random points as initial clusters' center
//    val kPoints = seed.shuffle(points.toList).take(K).toArray
    val kPoints = parsedData.takeSample(false, K, 42)

    val startTime = System.nanoTime

    var clusters : RDD[Cluster] = sc.parallelize(kPoints.zipWithIndex.map{
      case (k_p, id) => new Cluster(id, 1.0 / K, DenseVector(Array(k_p(0), k_p(1))), DenseMatrix.eye[Double](2))
    }).cache()

    val pointsM = new DenseMatrix(2, parsedData.count().toInt, parsedData.flatMap(a => List(a(0), a(1))).collect())

    println("Starting Centroids: ")
    printCentroids(clusters.collect(), scaleX, scaleY)

    // use maxIter as stopping criteria
    val oldLikelihood = 0.0
    var iter = 0
    var likelihood = 0.0
    var gaussians = new Array[Array[Double]](0)
    var gamma_nk = DenseMatrix.zeros[Double](1,1)
    var totals = DenseVector[Double](0)
    var gaussians_norm = new Array[Double](0)
    var gamma_nk_norm = DenseMatrix.zeros[Double](1,1)

    while (iter < maxIter) {
      // training step
//      val res = expMaxStep(sc, pointsM, clusters, K)
//      clusters = res._1
//      likelihood = res._2

      val startTime = System.nanoTime

      gaussians = clusters.map(a => a.gaussian(pointsM, isParallel = false)).collect()

      val duration = (System.nanoTime - startTime) / 1e9d
      println("gaussians: " + duration)

      gamma_nk = new DenseMatrix(pointsM.cols, K, gaussians.flatten)
      totals = sum(gamma_nk(*, ::))

      gaussians_norm = gaussians.flatMap(diag => (DenseVector(diag) / totals).toArray)
      gamma_nk_norm = new DenseMatrix(pointsM.cols, K, gaussians_norm).t

      // Maximization step
      clusters = clusters.map(_.maximizationStep(pointsM, gamma_nk_norm))
      clusters.persist()

      val sampleLikelihood = log(totals)
      likelihood = sum(sampleLikelihood)

      System.gc()

      println("Epoch: " + (iter + 1) + ", Likelihood: " + likelihood)
      iter += 1
    }

    // when training finishes, display duration time and clusters centroids
    val duration = (System.nanoTime - startTime) / 1e9d
    println()
    println("Sequential GMM duration:")
    println(duration)

    println()
    println("Final center points:")
    printCentroids(clusters.collect(), scaleX, scaleY)

    println("Iterations:")
    println(iter)

    println("Likelihood:")
    println(likelihood)
  }
}

// Python GMM cluster centers
//-----------------------------
//[[ 32.46567393  38.872867  ]
// [101.06939461  76.44165517]
// [ 11.99569006  16.10913821]
// [ 65.38825768 119.80146389]
// [249.64211625 226.74700378]]

// Scala GMM cluster centers
//-----------------------------
// Duration: 85.1374824
//
//Final center points:
//(11.300370175749158,15.273624932129978)
//(30.362547807896675,36.04434976921901)
//(60.72740631795588,124.44528524103906)
//(89.04657671648589,73.37369255558528)
//(242.03559485753297,219.53205201237546)