import ClusteringUtils.{import_files, minmax}
import java.io.{BufferedWriter, File, FileWriter}


object MinMax {

  def writeFile(filename: String, lines: Array[(Double, Double)]): Unit = {
    val file = new File(filename)
    val bw = new BufferedWriter(new FileWriter(file))
    for (line <- lines) {
      bw.write(line._1.toString + " " + line._2.toString + "\n")
    }
    bw.close()
  }

  def main(args: Array[String]): Unit = {
    val filename = "datasets/dataset_" + args(0) + ".txt"
    val (scaleX, scaleY, points) = minmax(import_files(filename))
    val newFileName = "datasets/dataset_" + args(0) + "_scaled.txt"
    val newFileNameScales = "datasets/scales_" + args(0) + ".txt"
    writeFile(newFileName, points)
    writeFile(newFileNameScales, Array(scaleX, scaleY))

  }
}

