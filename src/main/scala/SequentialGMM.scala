import ClusteringUtils._
import breeze.linalg.{DenseMatrix, DenseVector, _}
import breeze.numerics.log

import scala.annotation.tailrec


object SequentialGMM {

  def expMaxStep (points: DenseMatrix[Double], clusters : Array[Cluster]):
  (Array[Cluster], DenseMatrix[Double], Double) = {

    // Expectation step
    val gamma_nk = DenseMatrix.zeros[Double](points.cols, clusters.length)

    for (i <- 0 until gamma_nk.cols) {
      gamma_nk(::, i) := DenseVector(clusters(i).gaussian(points, isParallel = false))
    }

    val totals = sum(gamma_nk(*, ::))
    val gamma_nk_norm = DenseMatrix(
      (for (i <- 0 until gamma_nk.cols)
        yield gamma_nk(::, i) / totals) :_*)


    // Maximization step
    val newClusters = new Array[Cluster](clusters.length)
    for (i <- clusters.indices) {
      newClusters(i) = clusters(i).maximizationStep(points, gamma_nk_norm)
    }

    val sampleLikelihood = log(totals)
    val likelihood = sum(sampleLikelihood)

    (newClusters, gamma_nk_norm, likelihood)
  }


  def main(args: Array[String]): Unit = {

    // number of clusters
    val K = args(2).toInt

    val filename = "datasets/dataset_" + args(3) + "_scaled.txt"
    val scalesFilename = "datasets/scales_" + args(3) + ".txt"

    val (maxIter, tolerance, seed) = getHyperparameters()

    val points = import_files(filename)
    val scales = import_files(scalesFilename)
    val scaleX = scales(0)
    val scaleY = scales(1)

    println("Number of points:")
    println(points.length)

    // take 5 random points as initial clusters' center
    val kPoints = seed.shuffle(points.toList).take(K).toArray

    val startTime = System.nanoTime

    val clusters : Array[Cluster] = kPoints.zipWithIndex.map{
      case (k_p, id) => new Cluster(id, 1.0 / K, DenseVector(Array(k_p._1, k_p._2)), DenseMatrix.eye[Double](2))
    }

    val pointsM = new DenseMatrix(2, points.length, points.flatMap(a => List(a._1, a._2)))

    println("Starting Centroids: ")
    printCentroids(clusters, scaleX, scaleY)

    // use maxIter as stopping criteria
    val oldLikelihood = 0.0

    @tailrec
    def training(iter: Int, currentLikelihood: Double, currentClusters: Array[Cluster]): Unit ={
      if (iter >= maxIter) {
        // when training finishes, display duration time and clusters centroids
        val duration = (System.nanoTime - startTime) / 1e9d
        println()
        println("Sequential GMM duration:")
        println(duration)

        println()
        println("Final center points:")
        printCentroids(currentClusters, scaleX, scaleY)

        println("Iterations:")
        println(iter)

        println("Likelihood:")
        println(currentLikelihood)

      }
      else {
        // training step
        val (newClusters, gammaNK, likelihood) = expMaxStep(pointsM, currentClusters)
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

//Sequential GMM duration: 128.7709262
//
//Final center points:
//(11.300370175749158,15.273624932129978)
//(30.362547807896675,36.04434976921901)
//(60.72740631795588,124.44528524103906)
//(89.04657671648589,73.37369255558528)
//(242.03559485753297,219.53205201237546)
