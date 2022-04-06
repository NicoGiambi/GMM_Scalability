import scala.io.Source
import scala.sys.exit
import scala.util.Random

object SequentialKmeans {

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

  def main(args: Array[String]): Unit = {

    // K is the number of means (center points of clusters) to find
    val K = args(2).toInt

    // The device status data file(s)
    val filename = "datasets/dataset_" + args(3) + ".txt"

    // ConvergeDist -- the threshold "distance" between iterations at which we decide we are done
    val convergeDist = 1e-4
    val maxIter = 100
    // Parse the device status data file into pairs

    val startTime = System.nanoTime

    val points : Array[(Double, Double)] = import_files(filename)

//    points.foreach{ list => println("[" + list._1.toString + "," + list._2.toString + "]")}
//    println(points.length)

    for ((a, b) <- points.take(10)) {

      println("Width: " + a + " Height : " + b)

    }

    //start with K randomly selected points from the dataset as center points

    val kPoints = Random.shuffle(points.toList).take(K).toArray

    println("K Center points initialized :")

    for ((a, b) <- kPoints) {

      println("Width: " + a + " Height : " + b)

    }

    // loop until the total distance between one iteration's points and the next is less than the convergence distance specified
    var tempDist = Double.PositiveInfinity
    var iter = 0

    while (tempDist > convergeDist && iter < maxIter) {

      // For each key (k-point index), find a new point by calculating the average of each closest point

      // for each point, find the index of the closest kpoint.
      // map to (index, (point,1)) as follow:
      // (1, ((2.1,-3.4),1))
      // (0, ((5.1,-7.4),1))
      // (1, ((8.1,-4.4),1))

      val closestToKpoint = points.map(point => (closestPoint(point, kPoints), (point, 1)))

      // For each key (k-point index), reduce by sum (addPoints) the latitudes and longitudes of all the points closest to that k-point, and the number of closest points
      // E.g.
      // (1, ((4.325,-5.444),2314))
      // (0, ((6.342,-7.532),4323))
      // The reduced RDD should have at most K members.

      //val pointCalculatedRdd = closestToKpointRdd.reduceByKey((v1, v2) => ((addPoints(v1._1, v2._1), v1._2 + v2._2)))

      val pointCalculated = closestToKpoint.groupBy(_._1).mapValues(_.reduce(
        (p1, p2) =>
          (p1._1, (addPoints(p1._2._1, p2._2._1), p1._2._2 + p2._2._2))
      ))

      // For each key (k-point index), find a new point by calculating the average of each closest point
      // (index, (totalX,totalY),n) to (index, (totalX/n,totalY/n))

      //val newPointRdd = pointCalculatedRdd.map(center => (center._1, (center._2._1._1 / center._2._2, center._2._1._2 / center._2._2))).sortByKey()
      val newPoints = pointCalculated.map { case (k, (i, (point, n))) => (i, (point._1 / n, point._2 / n)) }

      // calculate the total of the distance between the current points (kPoints) and new points (localAverageClosestPoint)

      tempDist = 0.0

      for (i <- 0 until K) {
        // That distance is the delta between iterations. When delta is less than convergeDist, stop iterating
        tempDist += distanceSquared(kPoints(i), newPoints(i))
      }

      println("Distance between iterations (" + iter + "): " + tempDist)

      // Copy the new points to the kPoints array for the next iteration

      for (i <- 0 until K) {

        kPoints(i) = newPoints(i)

      }
//
//      // Display the final center points
//      println("Final center points :");
//
//      for (point <- kPoints) {
//        println(point);
//      }

//      // take 10 randomly selected device from the dataset and recall the model
//      val device = points.filter(device => !((device._1 == 0) && (device._2 == 0)))
//
//      val pointsRecall = Random.shuffle(device.toList).take(10)
//
//      for (point <- pointsRecall) {
//
//        val k = closestPoint(point, kPoints)
//        println("(W: " + point._1  + ", H: " + point._2 + ") to K: " + k);
//
//      }

      iter = iter + 1
    }

    val duration = (System.nanoTime - startTime) / 1e9d
    println()
    println("Sequential KMeans duration: " + duration)

    // Display the final center points
    println()
    println("Final center points :")
    kPoints.sortWith(_._1 < _._1).foreach(println)
  }
}

//----------------------------------------
//
//Dataset 16
//
//Sequential KMeans duration: 803.524115
//
//(26.137291649957294,29.359248393887267)
//(89.69036500359098,104.72585619914885)
//(153.18251212097312,268.5027359868489)
//(267.01983317250944,144.98912633666376)
//(439.7461952088592,338.62359696855435)

//----------------------------------------
//
//Dataset 1
//
//Sequential KMeans duration: 7.9322506
//
//(26.123173931806107,29.35010898759617)
//(89.63722708226182,104.59604608499957)
//(153.30727568381795,267.7204728172629)
//(266.90658791804753,145.02937803805727)
//(438.74359826474364,338.23832767594905)