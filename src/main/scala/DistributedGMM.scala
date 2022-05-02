import ClusteringUtils._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD

import scala.math.pow


object DistributedGMM {

  def det(matrix: DenseMatrix): Double = {
    matrix(0,0) * matrix(1,1) - matrix(0,1) * matrix(1,0)
  }

  def inv(matrix: DenseMatrix, det:Double) : DenseMatrix = {
    val elements = Array(matrix(1,1), -matrix(0,1), -matrix(1,0), matrix(0,0)).map(_ / det)
    new DenseMatrix(2, 2, elements)
  }

  def gaussian(sc: SparkContext, points: RDD[Vector], pi_k : Double, mu : DenseVector, cov : DenseMatrix): Array[Double] = {
    val diffVec = points.map(p => (p(0) - mu(0), p(1) - mu(1))).collect().flatMap(a => List(a._1, a._2))
    val diff = new DenseMatrix(2, points.count().toInt, diffVec)
    val pi = math.Pi
    val detCov = det(cov)
    val g1 = 1 / (pow(2 * pi, mu.size / 2) * math.sqrt(detCov))
    val dot = diff.transpose.multiply(inv(cov, detCov))
    // Non standard gaussian computation. The standard way needs too much memory, so we optimized it to run locally
    val diag = for (i <- 0 until dot.numRows)
      yield dot.rowIter.next().dot(diff.colIter.next())
    val gauss = diag.map(el => g1 * math.exp(-0.5 * el) * pi_k).toArray

    gauss
  }

  def expMaxStep (sc: SparkContext, points: RDD[Vector], clusters : Array[(Int, Double, DenseVector, DenseMatrix)]):
  (Array[(Int, Double, DenseVector, DenseMatrix)], DenseMatrix, Double) = {


    // Expectation step
    val gamma_nk = new DenseMatrix(points.count().toInt, clusters.length,
      (for (c <- clusters)
        yield gaussian(sc, points, c._2, c._3, c._4)).flatMap(_.toList))

    val totals = for (i <- 0 until gamma_nk.numRows)
      yield gamma_nk.colIter.next().toArray.sum

    val gamma_nk_norm = new DenseMatrix(points.count().toInt, clusters.length,
      (for (i <- 0 until gamma_nk.numCols)
        yield gamma_nk.rowIter.next().toArray.map(_ / totals(i))).flatMap(_.toList).toArray)

    // Maximization step

    val newClusters = for (c <- clusters) yield
    {
      val N = points.count().toInt

      val gammaRow = for (id <- 0 until N)
                      yield gamma_nk_norm(id, c._1)

      val N_k = gammaRow.sum
      val newPi = N_k / N
      val mu_k = points.zipWithIndex.map{
        case (p, id) => (p(0) * gammaRow(id.toInt) / N_k , p(1) * gammaRow(id.toInt) / N_k)
      }.reduce(addPoints)

      val newMu = new DenseVector(Array(mu_k._1, mu_k._2))

      val newDiffGamma = new DenseMatrix(2, points.count().toInt, points.zipWithIndex.map{
        case (p, id) =>
          (gammaRow(id.toInt) * (p(0) - newMu(0)), gammaRow(id.toInt) * (p(1) - newMu(1)))
      }.flatMap(a => List(a._1, a._2)).collect())

      val newDiff = new DenseMatrix(2, points.count().toInt, points.map(
        p => (p(0) - newMu(1), p(1) - newMu(1))
      ).flatMap(a => List(a._1, a._2)).collect()).transpose

      val newCovN = newDiffGamma.multiply(newDiff)
      val newCov = new DenseMatrix(2, 2, Array(newCovN(0,0) / N_k, newCovN(0,1) / N_k, newCovN(1,0) / N_k, newCovN(1,0) / N_k))
      (c._1, newPi, newMu, newCov)
    }

    val sampleLikelihood = totals.map(scala.math.log)
    val likelihood = sampleLikelihood.sum

    (newClusters, gamma_nk_norm, likelihood)
  }


  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    val conf = new SparkConf().setMaster(args(0)).setAppName("Benchmark")
    val sc = new SparkContext(conf)

    // K is the number of means (center points of clusters) to find
    val K = args(2).toInt

    // The device status data file(s)
    val filename = "datasets/dataset_" + args(3) + "_scaled.txt"
    val scalesFilename = "datasets/scales_" + args(3) + ".txt"

    val (maxIter, tolerance, seed) = getHyperparameters()

    val scales = import_files(scalesFilename)
    val data = sc.textFile(filename)
    val points = data.map(s => Vectors.dense(s.trim.split(' ').map(_.toDouble))).cache()
    val scaleX = scales(0)
    val scaleY = scales(1)

    println(points.count())

    val kPoints = points.takeSample(withReplacement = false, K, 42)

    val startTime = System.nanoTime

    var clusters = kPoints.zipWithIndex.map{
      case (k_p, id) => (id, 1.0 / K, new DenseVector(Array(k_p(0), k_p(1))), DenseMatrix.eye(2))
    }

    println("Starting Centroids: ")
    clusters.map(_._3).map(p => ((p(0) * (scaleX._2 - scaleX._1)) + scaleX._1, (p(1) * (scaleY._2 - scaleY._1)) + scaleY._1)).foreach(println)

    // loop until the total distance between one iteration's points and the next is less than the convergence distance specified
    var tempDist = Double.PositiveInfinity
    var iter = 0
    var oldLikelihood = 0.0
    while (iter < maxIter) {

      val (newClusters, gammaNK, likelihood) = expMaxStep(sc, points, clusters)
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
    //    clusters.map(_._4).foreach(println)
  }
}

//----------------------------------------
//
//Dataset 1
//---------------------------------------
//Sequential GMM duration: 283.2750701
//
//Final center points :
//(11.300370175748812,15.273624932130692)
//(30.36254780789383,36.04434976922017)
//(60.72740631795729,124.44528524100927)
//(89.04657671649437,73.37369255558039)
//(242.03559485754974,219.53205201237986)

//------------------------------------
//GMM duration: 218.6150163
//
//  [10.054827047125714,13.747923135172167]
//  [24.81223778368546,30.34074356942602]
//  [54.07884122770294,63.020182840268646]
//  [114.41900670826902,130.8535139210699]
//  [274.5198135230197,240.4293868991853]