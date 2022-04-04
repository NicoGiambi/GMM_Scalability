import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.{GaussianMixture, GaussianMixtureModel, KMeans}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import com.github.gradientgmm.GradientGaussianMixture
import java.nio.file.{Paths, Files}
import scala.io.Source
import scala.sys.exit
import org.apache.log4j.Logger
import org.apache.log4j.Level


object Benchmark {

  def import_files(path: String): Array[Float] = {
    val lines = Source.fromFile(path)
    val linesList = lines.getLines.toArray
    lines.close
    val toFloat: Array[Float] = linesList.map(_.toFloat)
    toFloat
  }

  def fitSave(model: String, sc: SparkContext, clusters: Int, outPath: String, parsedData: RDD[Vector]): Int = {

    val t1 = System.nanoTime

    if (model.equals("KMeans")) {
      val est = new KMeans().setK(clusters).setMaxIterations(100).run(parsedData)
      if (!Files.exists(Paths.get(outPath)))
        est.save(sc, outPath)
    }
    else if (model.equals("GMM")) {
      val est = new GaussianMixture().setK(clusters).setMaxIterations(100).run(parsedData)
      if (!Files.exists(Paths.get(outPath)))
        est.save(sc, outPath)
    }
    else if (model.equals("SGDGMM")) {
      val est = GradientGaussianMixture.fit(data = parsedData, k = clusters, kMeansIters = 0, kMeansTries = 0, maxIter = 100).toSparkGMM
      if (!Files.exists(Paths.get(outPath)))
        est.save(sc, outPath)
    }

    val duration1 = (System.nanoTime - t1) / 1e9d
    println(model + " duration: " + duration1)

    0
  }

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    val conf = new SparkConf().setMaster(args(0)).setAppName("DataImport")
    val sc = new SparkContext(conf)

    val model = args(1)
    val clusters = args(2).toInt

    val availableModels = Set("KMeans", "GMM", "SGDGMM")
    assert(availableModels.contains(args(1)))

    val outPath = "model/" + model + "/"

    val datasetPath = "datasets/dataset_" + args(3) + ".txt"
    val data = sc.textFile(datasetPath)
    val parsedData = data.map(s => Vectors.dense(s.trim.split(' ').map(_.toDouble))).cache()

    println("Fitting with " + args(0) + " on " + args(1) + " model with " + args(2) + " clusters and augmentation set to " + args(3))
    fitSave(model, sc, clusters, outPath, parsedData)

    println()
    println("------------------------------------")
    println()

    if (!availableModels.contains(model))
      println("No available models with this name")
    else if (model == "KMeans") {
      val estimator = KMeansModel.load(sc, outPath)
      val preds = estimator.predict(parsedData)
      if (!Files.exists(Paths.get(outPath + "/predictions")))
        preds.saveAsTextFile(outPath + "/predictions")
      estimator.clusterCenters.foreach(println)
    }
    else {
      val estimator = GaussianMixtureModel.load(sc, outPath)
      val preds = estimator.predict(parsedData)
      if (!Files.exists(Paths.get(outPath + "/predictions")))
        preds.saveAsTextFile(outPath + "/predictions")
      for (i <- 0 until estimator.k) {
        println(estimator.gaussians(i).mu)
      }
    }

    sc.stop()
  }
}
