import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix
import breeze.linalg._
import breeze.linalg.operators._

import scala.io.Source
import scala.math.exp
import scala.sys.exit
import scala.util.Random
import breeze.linalg.diag
import breeze.numerics.log
import breeze.stats.mean
import org.apache.log4j.{Level, Logger}

object SequentialGMM {

  import scala.math.pow

  def import_files(path: String): Array[(Double, Double)] = {
    val lines = Source.fromFile(path)
    val linesList = lines.getLines.toArray
    lines.close
    val couples = linesList.map(_.split(" +") match {
      case Array(s1, s2) => (s1.toDouble, s2.toDouble)
    })
    couples
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

    val scaledPoints = for (i <- points.indices) yield (scaledX(i), scaledY(i))
    ((xMin, xMax), (yMin, yMax), scaledPoints.toArray)
  }

  def gaussian(points: Array[(Double, Double)], pi_k : Double, mu : DenseVector[Double], cov : DenseMatrix[Double]): Array[Double] = {
    val diff = new DenseMatrix(2, points.length, points.map(p => (p._1 - mu(0), p._2 - mu(1))).flatMap(a => List(a._1, a._2)))
    val pi = math.Pi
    val g1 = 1 / (pow(2 * pi, mu.length / 2) * math.sqrt(det(cov)))
    val dot = diff.t * inv(cov)
    val diag = for (i <- 0 until diff.cols)
      yield dot(i, ::) * diff(::, i)
    val gauss = diag.map(el => g1 * math.exp(-0.5 * el) * pi_k).toArray

    gauss
  }

  def expMaxStep (points: Array[(Double, Double)], clusters : Array[(Int, Double, DenseVector[Double], DenseMatrix[Double])]):
    (Array[(Int, Double, DenseVector[Double], DenseMatrix[Double])], DenseMatrix[Double], Double) = {

    val gamma_nk = new DenseMatrix(points.length, clusters.length,
      (for (c <- clusters)
     yield gaussian(points, c._2, c._3, c._4)).flatMap(_.toList))

    val totals = sum(gamma_nk(*, ::))
    val gamma_nk_norm = DenseMatrix(
      (for (i <- 0 until gamma_nk.cols)
      yield gamma_nk(::, i) / totals) :_*)

    val newClusters = for (c <- clusters) yield
    {
      val N = points.length
      val gammaRow = gamma_nk_norm(c._1, ::)
      val N_k = sum(gammaRow)
      val newPi = N_k / N
      val mu_k = points.zipWithIndex.map{
        case (p, id) => (p._1 * gammaRow(id) , p._2 * gammaRow(id))
        }.reduce(addPoints)
      val newMu = DenseVector(Array(mu_k._1 / N_k, mu_k._2 / N_k))
      val newDiffGamma = new DenseMatrix(2, points.length, points.zipWithIndex.map{
        case (p, id) =>
          (gammaRow(id) * (p._1 - newMu(0)), gammaRow(id) * (p._2 - newMu(1)))
      }.flatMap(a => List(a._1, a._2)))

      val newDiff = new DenseMatrix(2, points.length, points.map(
        p => (p._1 - newMu(0), p._2 - newMu(1))
      ).flatMap(a => List(a._1, a._2))).t

      val newCov = (newDiffGamma * newDiff) / N_k
      (c._1, newPi, newMu, newCov)
    }

    val sampleLikelihood = log(totals)
    val likelihood = sum(sampleLikelihood)

    (newClusters, gamma_nk_norm, likelihood)
  }


  def main(args: Array[String]): Unit = {

    // K is the number of means (center points of clusters) to find
    val K = args(2).toInt

    // The device status data file(s)
    val filename = "datasets/dataset_" + args(3) + ".txt"

    val (scaleX, scaleY, points) = minmax(import_files(filename))
    println(points.length)

    val startTime = System.nanoTime

    // ConvergeDist -- the threshold "distance" between iterations at which we decide we are done
    val convergeDist = 1e-4 / max(scaleX._2, scaleY._2)
    val maxIter = 100
    // Parse the device status data file into pairs


    //    val points = minmax(Array((0.05, 1.413), (0.85, -0.3), (11.1, 0.4), (0.27, 0.12), (88, 12.33)))

    val kPoints = Random.shuffle(points.toList).take(K).toArray
//    val kPoints = points.take(K)

    var clusters : Array[(Int, Double, DenseVector[Double], DenseMatrix[Double])]= kPoints.zipWithIndex.map {
      case (k_p, id) => Tuple4(id, 1.0 / K, DenseVector(Array(k_p._1, k_p._2)), DenseMatrix.eye[Double](2))
    }

//    val mux = points.map(_._1).sum / points.length
//    val muy = points.map(_._2).sum / points.length
//    val mu = DenseVector(mux, muy)
//    val diff : DenseMatrix[Double] = new DenseMatrix(2, points.length, points.map(p => (p._1 - mu(0), p._2 - mu(1))).flatMap(a => List(a._1, a._2)))
//
//    val a = DenseMatrix((-1.0,0.0), (1.0,0.0), (-2.0,2.0), (3.0,2.0) ,(1.0,3.0))
//    val b = a.t * a
//    b.map(println)
//
//    val cov = diff * diff.t
//    val g = gaussian(points, mu, cov)
//
//
//    points.foreach(println)
//    println("------------------points")
//    mu.foreach(println)
//    println("------------------mu")
//    diff.map(println)
//    println("------------------diff")
//    cov.map(println)
//    println("------------------cov")
//    g.foreach(println)
//    exit(0)

    // loop until the total distance between one iteration's points and the next is less than the convergence distance specified
    var tempDist = Double.PositiveInfinity
    var iter = 0
    var oldLikelihood = 0.0
    while (iter < maxIter) {

      val (newClusters, gammaNK, likelihood) = expMaxStep(points, clusters)
      clusters = newClusters
//      tempDist = math.abs(likelihood - oldLikelihood)
//      println("Epoch: " + (iter + 1) + ", Likelihood: " + likelihood + ", Difference: " + tempDist)
      println("Epoch: " + (iter + 1) + ", Likelihood: " + likelihood)
      iter = iter + 1
      oldLikelihood = likelihood
    }

    val duration = (System.nanoTime - startTime) / 1e9d
    println()
    println("Sequential GMM duration: " + duration)

    // Display the final center points
    println()
    println("Final center points :")
    clusters.map(_._3).map(p => ((p(0) * (scaleX._2 - scaleX._1)) + scaleX._1, (p(1) * (scaleY._2 - scaleY._1)) + scaleY._1)).sortWith(_._1 < _._1).foreach(println)
  }
}

//----------------------------------------
//
//Dataset 1
//
//Sequential GMM duration: ?? 308.7006298
//
//(10.302604938500155,14.111168725629852)
//(26.0148436342939,31.646147918711506)
//(57.482978113970944,66.89993006257836)
//(121.18373122223119,138.09760439894762)
//(283.5337149247836,244.98232181715147)