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


  def run(parsedData: Array[(Double, Double)],
          kPoints: Array[(Double, Double)],
          scales: Array[(Double, Double)],
          K: Int,
          maxIter: Int): Unit = {

    val scaleX = scales(0)
    val scaleY = scales(1)

    val clusters : Array[Cluster] = kPoints.zipWithIndex.map{
      case (k_p, id) => new Cluster(id, 1.0 / K, DenseVector(Array(k_p._1, k_p._2)), DenseMatrix.eye[Double](2))
    }

    val pointsM = new DenseMatrix(2, parsedData.length, parsedData.flatMap(a => List(a._1, a._2)))

    println("Starting Centroids: ")
    printCentroids(clusters, scaleX, scaleY)

    // use maxIter as stopping criteria
    val oldLikelihood = 0.0

    @tailrec
    def training(iter: Int, currentLikelihood: Double, currentClusters: Array[Cluster]): Unit ={

      if (iter >= maxIter) {
        // when training finishes, display duration time and clusters centroids
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
        val startTime = System.nanoTime
        val (newClusters, gammaNK, likelihood) = expMaxStep(pointsM, currentClusters)
        val duration = (System.nanoTime - startTime) / 1e9d

        println("Epoch: " + (iter + 1) + ", Likelihood: " + likelihood + ", Duration: " + duration)
        training(iter + 1, likelihood, newClusters)

      }
    }

    training(0, oldLikelihood, clusters)

  }

  def main(args: Array[String]): Unit = {

    val filename = args(4) + "dataset_" + args(3) + "_scaled.txt"
    val scalesFilename = args(4) + "scales_" + args(3) + ".txt"

    val (maxIter, tolerance, seed) = getHyperparameters()

    val points = import_files(filename)
    val scales = import_files(scalesFilename)
    val K = args(2).toInt

    println("Number of points:")
    println(points.length)

    // take 5 random points as initial clusters' center
    val kPoints = seed.shuffle(points.toList).take(K).toArray

    val startTime = System.nanoTime()

    run(parsedData = points,
        scales = scales,
        kPoints = kPoints,
        K = K,
        maxIter = maxIter)

    val duration = (System.nanoTime - startTime) / 1e9d
    println("sequentialGMM duration: " + duration)
  }

}

