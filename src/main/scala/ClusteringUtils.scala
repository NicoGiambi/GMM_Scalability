import breeze.linalg.{*, DenseMatrix, DenseVector, det, inv, sum}
import org.apache.spark.mllib.clustering.GaussianMixtureModel

import scala.io.Source
import scala.math.pow


class Cluster(val id: Int, val pi_k: Double, val center: DenseVector[Double], val covariance: DenseMatrix[Double]) extends Serializable {

  def computeDiagonalPar(diff: DenseMatrix[Double], dot: DenseMatrix[Double], g1: Double): Array[Double] = {
    val diag = for (i <- (0 until diff.cols).par)
      yield dot(i, ::) * diff(::, i)
    diag.par.map(el => g1 * math.exp(-0.5 * el) * pi_k).toArray
  }

  def computeDiagonal(diff: DenseMatrix[Double], dot: DenseMatrix[Double], g1: Double): Array[Double] = {
    val diag = for (i <- 0 until diff.cols)
      yield dot(i, ::) * diff(::, i)
    diag.map(el => g1 * math.exp(-0.5 * el) * pi_k).toArray
  }

  // Compute the gaussian relative to each cluster
  def gaussian(points: DenseMatrix[Double], isParallel: Boolean): Array[Double] = {
    val diff = points.copy
    diff(0, ::) :-= center(0)
    diff(1, ::) :-= center(1)

    val pi = math.Pi
    val g1 = 1 / (pow(2 * pi, center.length / 2) * math.sqrt(det(covariance)))
    val dot = diff.t * inv(covariance)

    // Non standard gaussian computation. The standard way needs too much memory, so we optimized it to run locally
    if (isParallel)
      computeDiagonalPar(diff, dot, g1)
    else
      computeDiagonal(diff, dot, g1)
  }


  def maximizationStep(points: DenseMatrix[Double], gamma_nk_norm: DenseMatrix[Double]): Cluster ={
    val N = points.cols
    val gammaRow = gamma_nk_norm(id, ::)
    val N_k = sum(gammaRow)
    val newPi = N_k / N

    val weightedPoints = points.copy
    weightedPoints(0, ::) :*= gammaRow / N_k
    weightedPoints(1, ::) :*= gammaRow / N_k

    val newMu = DenseVector(sum(weightedPoints(*, ::)).toArray)

    val newDiffGamma = points.copy
    newDiffGamma(0, ::) :-= newMu(0)
    newDiffGamma(1, ::) :-= newMu(1)

    val newDiff = newDiffGamma.copy.t

    newDiffGamma(0, ::) :*= gammaRow
    newDiffGamma(1, ::) :*= gammaRow

    val newCov = (newDiffGamma * newDiff) / N_k

    val newCluster = new Cluster(id, newPi, newMu, newCov)

    newCluster
  }
}

object ClusteringUtils {

  def getHyperparameters: (Int, Double) = {
    val maxIter = 50    // We saw that after 50 iterations the likelihood is quite stable
    val tolerance = 0   // We set the tolerance to 0 to test the effective duration of each algorithm
    (maxIter, tolerance)
  }

  def printCentroids(clusters: Array[Cluster],
                     scaleX: (Double, Double),
                     scaleY: (Double, Double)): Unit ={
    clusters
      .map(_.center)
      .map(p => ((p(0) * (scaleX._2 - scaleX._1)) + scaleX._1, (p(1) * (scaleY._2 - scaleY._1)) + scaleY._1))
      .sortWith(_._1 < _._1)
      .foreach(println)
    println()
  }

  def printMllibCentroids(estimator: GaussianMixtureModel,
                          scaleX: (Double, Double),
                          scaleY: (Double, Double)): Unit ={
    estimator
      .gaussians
      .map(_.mu)
      .map(p => ((p(0) * (scaleX._2 - scaleX._1)) + scaleX._1, (p(1) * (scaleY._2 - scaleY._1)) + scaleY._1))
      .sortWith(_._1 < _._1)
      .foreach(println)
    println()
  }

  def import_files(path: String): Array[(Double, Double)] = {
    val lines = Source.fromFile(path)
    val linesList = lines.getLines.toArray
    lines.close
    val couples = linesList.par.map(_.split(" +") match {
      case Array(s1, s2) => (s1.toDouble, s2.toDouble)
    })
    couples.toArray
  }

  // Scales the dataset between [0, 1]
  def minmax(points : Array[(Double, Double)]): ((Double, Double), (Double, Double), Array[(Double, Double)]) ={
    val flatX = points.map(_._1)
    val xMin = flatX.min
    val xMax = flatX.max
    val scaledX = flatX.map(p => (p - xMin) / (xMax - xMin))
    val flatY = points.map(_._2)
    val yMin = flatY.min
    val yMax = flatY.max
    val scaledY = flatY.map(p => (p - yMin) / (yMax - yMin))

    val scaledPoints = scaledX zip scaledY
    ((xMin, xMax), (yMin, yMax), scaledPoints)
  }

}
