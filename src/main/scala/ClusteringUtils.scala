import breeze.linalg.{DenseMatrix, DenseVector, det, inv, sum}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Matrix, Matrices}

import scala.io.Source
import scala.math.pow
import scala.util.Random


class Cluster(val id: Int, val pi_k: Double, val center: DenseVector[Double], val covariance: DenseMatrix[Double]) {
  def gaussian(points: Array[(Double, Double)]): Array[Double] = {
    val diff = new DenseMatrix(2, points.length, points.map(p => (p._1 - center(0), p._2 - center(1))).flatMap(a => List(a._1, a._2)))
    val pi = math.Pi
    val g1 = 1 / (pow(2 * pi, center.length / 2) * math.sqrt(det(covariance)))
    val dot = diff.t * inv(covariance)

    // Non standard gaussian computation. The standard way needs too much memory, so we optimized it to run locally
    val diag = for (i <- 0 until diff.cols)
      yield dot(i, ::) * diff(::, i)
    val gauss = diag.map(el => g1 * math.exp(-0.5 * el) * pi_k).toArray

    gauss
  }

//  def gaussianRDD(points: RDD[(Double, Double)], length: Int): Int = {
//    val diff = new DenseMatrix(2, length, points.map(p => (p._1 - center(0), p._2 - center(1))).flatMap(a => List(a._1, a._2)))
//    val pi = math.Pi
//    val g1 = 1 / (pow(2 * pi, center.length / 2) * math.sqrt(det(covariance)))
//    val dot = diff.t * inv(covariance)
//
//    // Non standard gaussian computation. The standard way needs too much memory, so we optimized it to run locally
//    val diag = for (i <- 0 until diff.cols)
//      yield dot(i, ::) * diff(::, i)
//    val gauss = diag.map(el => g1 * math.exp(-0.5 * el) * pi).toArray
//
//    gauss
//  }

  def maximizationStep(points: Array[(Double, Double)], gamma_nk_norm: DenseMatrix[Double]): Array[Cluster] ={
    val N = points.length
    val gammaRow = gamma_nk_norm(id, ::)
    val N_k = sum(gammaRow)
    val newPi = N_k / N
    val mu_k = points.zipWithIndex.map {
      case (p, id) => (p._1 * gammaRow(id), p._2 * gammaRow(id))
    }.reduce(ClusteringUtils.addPoints)
    val newMu = DenseVector(Array(mu_k._1 / N_k, mu_k._2 / N_k))

    val newDiffGamma = new DenseMatrix(2, points.length, points.zipWithIndex.map {
      case (p, id) =>
        (gammaRow(id) * (p._1 - newMu(0)), gammaRow(id) * (p._2 - newMu(1)))
    }.flatMap(a => List(a._1, a._2)))

    val newDiff = new DenseMatrix(2, points.length, points.map(
      p => (p._1 - newMu(0), p._2 - newMu(1))
    ).flatMap(a => List(a._1, a._2))).t

    val newCov = (newDiffGamma * newDiff) / N_k

    val newCluster = Array(new Cluster(id, newPi, newMu, newCov))

    newCluster
  }
}

object ClusteringUtils {

  def import_files(path: String): Array[(Double, Double)] = {
    val lines = Source.fromFile(path)
    val linesList = lines.getLines.toArray
    lines.close
    val couples = linesList.map(_.split(" +") match {
      case Array(s1, s2) => (s1.toDouble, s2.toDouble)
    })
    couples
  }

  def getHyperparameters (): (Int, Double, Random) = {
    val maxIter = 100
    val tolerance = 1e-4
    val randomSeed = new Random(42)
    (maxIter, tolerance, randomSeed)
  }

  def printCentroids(clusters: Array[Cluster],
                     scaleX: (Double, Double),
                     scaleY: (Double, Double)): Unit ={
    clusters
      .map(_.center)
      .map(p => ((p(0) * (scaleX._2 - scaleX._1)) + scaleX._1, (p(1) * (scaleY._2 - scaleY._1)) + scaleY._1))
      .sortWith(_._1 < _._1)
      .foreach(println)
  }

  // The squared distances between two points
  def distanceSquared(p1: (Double, Double), p2: (Double, Double)) : Double = {
    pow(p1._1 - p2._1, 2) + pow(p1._2 - p2._2, 2)
  }

  // The sum of two points
  def addPoints(p1: (Double, Double), p2: (Double, Double)) : (Double, Double) = {
    (p1._1 + p2._1, p1._2 + p2._2)
  }

  // for a point p and an array of points, return the index in the array of the point closest to p
  def closestPoint(p: (Double, Double), kpoints: Array[(Double, Double)]): Int = {
    var index = 0
    var bestIndex = 0
    var closest = Double.PositiveInfinity

    for (i <- kpoints.indices) {
      val dist = distanceSquared(p, kpoints(i))
      if (dist < closest) {
        closest = dist
        bestIndex = i
      }
    }
    bestIndex
  }

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
