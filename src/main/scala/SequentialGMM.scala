import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix
import breeze.linalg._
import ClusteringUtils._
import breeze.numerics.log
import scala.math.pow


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

    // The device status data file(s)
//    val filename = "datasets/dataset_" + args(3) + "_scaled.txt"
//    val scalesFilename = "datasets/scales_" + args(3) + ".txt"
    val filename = "datasets/dataset_" + args(3) + ".txt"
    val (maxIter, tolerance, seed) = getHyperparameters()

    val (scaleX, scaleY, points) = minmax(import_files(filename))
//    val points = import_files(filename)
//    val scales = import_files(scalesFilename)
//    val scaleX = scales(0)
//    val scaleY = scales(1)

    println("Number of points:")
    println(points.length)

    val kPoints = seed.shuffle(points.toList).take(K).toArray

    val startTime = System.nanoTime

    var clusters : Array[Cluster] = kPoints.zipWithIndex.map{
      case (k_p, id) => new Cluster(id, 1.0 / K, DenseVector(Array(k_p._1, k_p._2)), DenseMatrix.eye[Double](2))
    }

    println("Starting Centroids: ")
    printCentroids(clusters, scaleX, scaleY)

    // loop until the total distance between one iteration's points and the next is less than the convergence distance specified
    var tempDist = Double.PositiveInfinity
    val oldLikelihood = 0.0

    def training(iter: Int, currentLikelihood: Double, currentClusters: Array[Cluster]): Unit ={
      if (iter >= maxIter) {

        val duration = (System.nanoTime - startTime) / 1e9d
        println()
        println("Sequential GMM duration:")
        println(duration)

        // Display the final center points
        println()
        println("Final center points:")
        printCentroids(currentClusters, scaleX, scaleY)
        //    clusters.map(_._4).foreach(println)
        println("Iterations:")
        println(iter)

        println("Likelihood:")
        println(currentLikelihood)

      }
      else {
        val (newClusters, gammaNK, likelihood) = expMaxStep(points, currentClusters)
        tempDist = math.abs(likelihood - oldLikelihood)
        println("Epoch: " + (iter + 1) + ", Likelihood: " + likelihood + ", Difference: " + tempDist)
        training(iter + 1, likelihood, newClusters)
      }
    }

    training(0, oldLikelihood, clusters)
//    while (iter < maxIter) {
//
//      val (newClusters, gammaNK, likelihood) = expMaxStep(points, clusters)
//      clusters = newClusters
//      //      tempDist = math.abs(likelihood - oldLikelihood)
//      //      println("Epoch: " + (iter + 1) + ", Likelihood: " + likelihood + ", Difference: " + tempDist)
//      println("Epoch: " + (iter + 1) + ", Likelihood: " + likelihood)
//      iter = iter + 1
//      oldLikelihood = likelihood
//    }


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

// Python GMM cluster centers
//[[ 32.46567393  38.872867  ]
// [101.06939461  76.44165517]
// [ 11.99569006  16.10913821]
// [ 65.38825768 119.80146389]
// [249.64211625 226.74700378]]