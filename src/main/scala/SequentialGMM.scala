import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix
import breeze.linalg._
import ClusteringUtils._
import breeze.numerics.log

import scala.annotation.tailrec


object SequentialGMM {

  def expMaxStep (points: Array[(Double, Double)], clusters : Array[Cluster]):
                 (Array[Cluster], DenseMatrix[Double], Double) = {

    // Expectation step
    val gamma_nk = new DenseMatrix(points.length,
                                   clusters.length,
                                   clusters.flatMap(_.gaussian(points)))

    val totals = sum(gamma_nk(*, ::))
    val gamma_nk_norm = DenseMatrix(
      (for (i <- 0 until gamma_nk.cols)
        yield gamma_nk(::, i) / totals): _*)

    // Maximization step
    val newClusters = clusters.flatMap(_.maximizationStep(points, gamma_nk_norm))

    val sampleLikelihood = log(totals)
    val likelihood = sum(sampleLikelihood)

    (newClusters, gamma_nk_norm, likelihood)
  }


  def main(args: Array[String]): Unit = {

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

    val clusters : Array[Cluster] = kPoints.zipWithIndex.map{
      case (k_p, id) => new Cluster(id, 1.0 / K, DenseVector(Array(k_p._1, k_p._2)), DenseMatrix.eye[Double](2))
    }

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
        val (newClusters, gammaNK, likelihood) = expMaxStep(points, currentClusters)
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
// Duration: 1664.1321194
//-----------------------------
//(9.933946207734696,13.70182011905065)
//(22.940796982540448,28.244988984963488)
//(44.761694262831774,52.41992124688973)
//(90.3236292546855,103.52162824790214)
//(230.21380838793382,214.16115834720713)

// Scala Parallel GMM cluster centers
//-----------------------------
// Duration: 1301.3916283
//-----------------------------
//(9.933946207734696,13.70182011905065)
//(22.940796982540448,28.244988984963488)
//(44.761694262831774,52.41992124688973)
//(90.3236292546855,103.52162824790214)
//(230.21380838793382,214.16115834720713)