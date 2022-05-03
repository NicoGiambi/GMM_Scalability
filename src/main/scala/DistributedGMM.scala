import ClusteringUtils._
import breeze.linalg.{DenseMatrix, DenseVector, _}
import breeze.numerics.log
import org.apache.spark.rdd.RDD

import scala.annotation.tailrec
import org.apache.spark.{SparkConf, SparkContext}


object DistributedGMM {

  def expMaxStep (sc: SparkContext, points: DenseMatrix[Double], clusters : RDD[Cluster]):
  (RDD[Cluster], DenseMatrix[Double], Double) = {

    val K = clusters.count().toInt
    // Expectation step
    // distribute on columns, i.e. on clusters
    val gaussians = clusters.map(_.gaussian(points, isParallel = true))
    val gamma_nk = new DenseMatrix(points.cols, K, gaussians.flatMap(a => a).collect())
    val totals = sum(gamma_nk(*, ::))
    val gaussians_norm = gaussians.flatMap(diag => (DenseVector(diag) / totals).toArray)
    val gamma_nk_norm = new DenseMatrix(points.cols, K, gaussians_norm.collect()).t

    // Maximization step
    val newClusters = clusters.map(_.maximizationStep(points, gamma_nk_norm)(0))


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
    val filename = "datasets/dataset_" + args(3) + ".txt"
    val (maxIter, tolerance, seed) = getHyperparameters()

    // scale anchors dimension in range [0, 1]
    val (scaleX, scaleY, points) = minmax(import_files(filename))

    println("Number of points:")
    println(points.length)

    // take 5 random points as initial clusters' center
    val kPoints = seed.shuffle(points.toList).take(K).toArray

    val startTime = System.nanoTime

    val clusters : RDD[Cluster] = sc.parallelize(kPoints.zipWithIndex.map{
      case (k_p, id) => new Cluster(id, 1.0 / K, DenseVector(Array(k_p._1, k_p._2)), DenseMatrix.eye[Double](2))
    })

    val pointsM = new DenseMatrix(2, points.length, points.par.flatMap(a => List(a._1, a._2)).toArray)

    println("Starting Centroids: ")
//    printCentroids(clusters.collect(), scaleX, scaleY)

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
//        printCentroids(currentClusters.collect(), scaleX, scaleY)

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