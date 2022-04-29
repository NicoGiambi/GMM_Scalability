import scala.io.Source
import scala.math.pow
import scala.util.Random

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

  def getHyperparameters (): (Int, Double, Random) = {
    val maxIter = 100
    val tolerance = 0
    val randomSeed = new Random(42)
    (maxIter, tolerance, randomSeed)
  }

}
