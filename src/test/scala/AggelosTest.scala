import org.apache.spark.graphx.{Edge, Graph}
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, BlockMatrixFixedOps, CoordinateMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.junit.Test

import scala.io.Source
import scala.reflect.ClassTag
import scala.reflect.io.Path
import scala.util.Try

/**
  * Created by oraviv on 14/10/2016.
  */
class AggelosTest {

  val sparkConf = new SparkConf().setMaster("local[1]")
    .set("spark.sql.tungsten.enabled", "false")
    .set("spark.sql.shuffle.partitions", "2")
    //    .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .set("spark.ui.retainedJobs", "10000")
    .set("spark.ui.retainedStages", "10000")
    .set("spark.executor.extraJavaOptions", "-XX:+PrintGCDetails")
    .set("spark.driver.extraJavaOptions", "-XX:MaxPermSize=256m")
    .set("spark.local.dir", "/tmp/spark-test")
    .setAppName("Spark Unit Test")

  implicit val sc = new SparkContext(sparkConf)
  sc.setCheckpointDir("/tmp/spark-checkpoint")

  def removeOutput() = {
    val path: Path = Path ("src/test/test-output/multiply-result")
    Try(path.deleteRecursively())
  }

  removeOutput()

  @Test
  def multiply_test() = {
    val lines = sc.textFile("src/test/resources/data/myfile-2000000-1e-05").sample(false, 0.0001, 123)
    //val header = lines.first()
    //val footer= "--->8------>8------>8------>8------>8---"

    val pairs = lines.map(s => s.split(" "))
                     .map(s => MatrixEntry(s(0).toInt,s(1).toInt,s(2).toDouble))
                     .repartition(2)

    val mat: CoordinateMatrix = new CoordinateMatrix(pairs)
    val bMat = mat.toBlockMatrix(10000,10000)

    val resMat: BlockMatrix = BlockMatrixFixedOps.multiply(bMat,bMat.transpose)


    resMat.blocks.saveAsTextFile("src/test/test-output/multiply-result")

  }

  @Test
  def mcl_test() = {
    val g = ensureBiDiGraph(createGraphFromEdges(Source.fromURL(getClass.getResource("/graph/sample1/weighted_edges.txt")).getLines().toSeq))
    val assignments = MCL.run(g, epsilon=1.0/500, maxIterations=15, rePartitionMatrix = None, reIndexNodes = false, blockSize = 10000)
    assignments.collect().foreach(println)
  }

  def createGraphFromEdges(edgesLines: Seq[String], numPartitions: Int = 5)(implicit sc: SparkContext): Graph[String, Double] = {

    val edges: RDD[Edge[Double]] =
      sc.parallelize(
        edgesLines
          //.map(line => {println(line); line.split(" ")})
          .map(line => line.split(" "))
          .map(e => Edge(e(0).toLong, e(1).toLong, if (e.length>2) e(2).toDouble else 1.0))
      ).repartition(numPartitions)

    val graph: Graph[String, Double] = Graph.fromEdges(edges, "default")

    graph
  }

  def ensureBiDiGraph[VD: ClassTag](g: Graph[VD,Double])(implicit sc: SparkContext): Graph[String, Double] = {
    val tempEntries = g.triplets.flatMap(
      triplet => {
        Array(
          ((triplet.srcId, triplet.dstId), (triplet.attr, 1)),
          ((triplet.dstId, triplet.srcId), (triplet.attr, 2))
        )
      }
    )

    val newEdges: RDD[Edge[Double]] = tempEntries.groupByKey().map(
      e => Edge(e._1._1, e._1._2, e._2.map(_._1).max)
    )

    Graph.fromEdges(newEdges, "node")

  }
}
