import ClusteringUtils._
import breeze.linalg.{DenseMatrix, DenseVector, _}
import breeze.numerics.log
import scala.annotation.tailrec


object ParallelGMM {

  def expMaxStep (points: DenseMatrix[Double], clusters : Array[Cluster]):
                 (Array[Cluster], Double) = {

    // Expectation step - parallelize on columns, i.e. on clusters
    val gamma_nk = DenseMatrix.zeros[Double](points.cols, clusters.length)

    for (i <- (0 until gamma_nk.cols).par) {
      gamma_nk(::, i) := DenseVector(clusters(i).gaussian(points, isParallel = true))
    }

    val totals = sum(gamma_nk(*, ::))
    val gamma_nk_norm = DenseMatrix(
      (for (i <- (0 until gamma_nk.cols).par)
        yield gamma_nk(::, i) / totals).toIndexedSeq :_*)

    // Maximization step
    val newClusters = new Array[Cluster](clusters.length)
    for (i <- clusters.indices.par) {
      newClusters(i) = clusters(i).maximizationStep(points, gamma_nk_norm)
    }

    val sampleLikelihood = log(totals)
    val likelihood = sum(sampleLikelihood)

    (newClusters, likelihood)
  }

  // Same as SequentialGMM, but we use .par when possible
  def run(parsedData: Array[(Double, Double)],
          kPoints: Array[(Double, Double)],
          scales: Array[(Double, Double)],
          K: Int,
          maxIter: Int): Unit = {

    val scaleX = scales(0)
    val scaleY = scales(1)

    // initialize an Array of Clusters
    val clusters : Array[Cluster] = kPoints.zipWithIndex.par.map{
      case (k_p, id) => new Cluster(id = id,
                                    pi_k = 1.0 / K,
                                    center = DenseVector(Array(k_p._1, k_p._2)),
                                    covariance = DenseMatrix.eye[Double](2))
    }.toArray

    // dataset as matrix (2 x nPoints)
    val pointsM = new DenseMatrix(2, parsedData.length, parsedData.par.flatMap(a => List(a._1, a._2)).toArray)

    println("Starting Centroids: ")
    printCentroids(clusters, scaleX, scaleY)

    // tail recursion on the ExpMax step
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
        val (newClusters, likelihood) = expMaxStep(pointsM, currentClusters)
        val duration = (System.nanoTime - startTime) / 1e9d

        println("Epoch: " + (iter + 1) + ", Likelihood: " + likelihood + ", Duration: " + duration)
        training(iter + 1, likelihood, newClusters)

      }
    }

    training(0, 0.0, clusters)

  }
}
