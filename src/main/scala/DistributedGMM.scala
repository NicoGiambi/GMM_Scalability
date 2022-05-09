import ClusteringUtils._
import breeze.linalg.{DenseMatrix, DenseVector, _}
import breeze.numerics.log
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD

import scala.annotation.tailrec
import org.apache.spark.{SparkConf, SparkContext}


object DistributedGMM {

  def expMaxStep (sc: SparkContext, points: DenseMatrix[Double], clusters : RDD[Cluster]):
  (RDD[Cluster], DenseMatrix[Double], Double) = {

    val K = clusters.count().toInt
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
    val newClusters = clusters.map(_.maximizationStep(points, gamma_nk_norm)(0))

    val duration1 = (System.nanoTime - startTime1) / 1e9d
    println("maximization_step: " + duration1)



    val sampleLikelihood = log(totals)
    val likelihood = sum(sampleLikelihood)

    (newClusters, gamma_nk_norm, likelihood)
  }


  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("DistributedGMM").setMaster(args(0))
    val sc = new SparkContext(conf)

    // number of clusters
    val K = args(2).toInt

    // file with anchors sizes
    val filename = "datasets/dataset_" + args(3) + "_scaled.txt"
    val scalesFilename = "datasets/scales_" + args(3) + ".txt"

    val (maxIter, tolerance, seed) = getHyperparameters()

    val points = import_files(filename)
    val scales = import_files(scalesFilename)
    val data = sc.parallelize(points)
    val parsedData = data.map(s => Vectors.dense(Array(s._1, s._2))).cache()
    val scaleX = scales(0)
    val scaleY = scales(1)

    println("Number of points:")
    println(points.length)

    // take 5 random points as initial clusters' center
    val kPoints = seed.shuffle(points.toList).take(K).toArray

    val startTime = System.nanoTime

    val clusters : RDD[Cluster] = sc.parallelize(kPoints.zipWithIndex.map{
      case (k_p, id) => new Cluster(id, 1.0 / K, DenseVector(Array(k_p._1, k_p._2)), DenseMatrix.eye[Double](2))
    })

    val pointsM = new DenseMatrix(2, points.length, parsedData.flatMap(a => List(a(0), a(1))).collect())

    println("Starting Centroids: ")
    printCentroids(clusters.collect(), scaleX, scaleY)

    // use maxIter as stopping criteria
    val oldLikelihood = 0.0

    @tailrec
    def training(iter: Int, currentLikelihood: Double, currentClusters: RDD[Cluster]): Unit ={
      if (iter >= maxIter) {
        // when training finishes, display duration time and clusters centroids
        val duration = (System.nanoTime - startTime) / 1e9d
        println()
        println("Sequential GMM duration:")
        println(duration)

        println()
        println("Final center points:")
        printCentroids(currentClusters.collect(), scaleX, scaleY)

        println("Iterations:")
        println(iter)

        println("Likelihood:")
        println(currentLikelihood)

      }
      else {
        // training step
        val (newClusters, gammaNK, likelihood) = expMaxStep(sc, pointsM, currentClusters)
        println("Epoch: " + (iter + 1) + ", Likelihood: " + likelihood)
        training(iter + 1, likelihood, newClusters)
      }
    }

    training(0, oldLikelihood, clusters)

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