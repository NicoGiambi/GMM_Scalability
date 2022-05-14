import ClusteringUtils._
import breeze.linalg.{DenseMatrix, DenseVector, _}
import breeze.numerics.log
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext

object DistributedGMM {

  def run(sc: SparkContext,
          parsedData: RDD[Vector],
          kPoints: Array[(Double, Double)],
          scales: Array[(Double, Double)],
          K: Int,
          maxIter: Int): Unit = {

    val scaleX = scales(0)
    val scaleY = scales(1)

    // initialize an RDD of Clusters
    var clusters : RDD[Cluster] = sc.parallelize(kPoints.zipWithIndex.map{
      case (k_p, id) => new Cluster(id = id,
                                    pi_k = 1.0 / K,
                                    center = DenseVector(Array(k_p._1, k_p._2)),
                                    covariance = DenseMatrix.eye[Double](2))
    }).cache()

    // dataset as matrix (2 x nPoints)
    val pointsM = new DenseMatrix(2, parsedData.count().toInt, parsedData.flatMap(a => List(a(0), a(1))).collect())

    println("Starting Centroids: ")
    printCentroids(clusters.collect(), scaleX, scaleY)

    // we expand the ExpectationStep in the main because .collect() exhaust heap space
    var iter = 0
    var likelihood = 0.0
    var gaussians = new Array[Array[Double]](0)
    var gamma_nk = DenseMatrix.zeros[Double](1,1)
    var totals = DenseVector[Double](0)
    var gaussians_norm = new Array[Double](0)
    var gamma_nk_norm = DenseMatrix.zeros[Double](1,1)
    var startTime = 0.0
    var duration = 0.0

    while (iter < maxIter) {

      startTime = System.nanoTime

      gaussians = clusters.map(a => a.gaussian(pointsM, isParallel = false)).collect()

      gamma_nk = new DenseMatrix(pointsM.cols, K, gaussians.flatten)
      totals = sum(gamma_nk(*, ::))

      gaussians_norm = gaussians.flatMap(diag => (DenseVector(diag) / totals).toArray)
      gamma_nk_norm = new DenseMatrix(pointsM.cols, K, gaussians_norm).t

      // Maximization step
      clusters = clusters.map(_.maximizationStep(pointsM, gamma_nk_norm))
      clusters.persist()

      val sampleLikelihood = log(totals)
      likelihood = sum(sampleLikelihood)

      duration = (System.nanoTime - startTime) / 1e9d

      println("Epoch: " + (iter + 1) + ", Likelihood: " + likelihood + ", Duration: " + duration)
      iter += 1
    }

    println()
    println("Final center points:")
    printCentroids(clusters.collect(), scaleX, scaleY)

    println("Iterations:")
    println(iter)

    println("Likelihood:")
    println(likelihood)
  }
}


