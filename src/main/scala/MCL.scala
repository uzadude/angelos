/*The MIT License (MIT)

Copyright (c) 2015-2016, Joan André

Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.*/

import org.apache.spark.Logging
import org.apache.spark.graphx._
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.rdd.RDD


/**
  * MCL clustering algorithm.
  *
  * Clusters the graph according to: http://micans.org/mcl/
  *
  * The resulting RDD[(VertexId, VertexId)] contains the vertices added with their respective cluster number.
  *
  */
object MCL {

  /** Train an MCL model using the given set of parameters.
    *
    * @param graph         training points stored as `BlockMatrix`
    * @param expansionRate expansion rate of adjacency matrix at each iteration
    * @param inflationRate inflation rate of adjacency matrix at each iteration
    * @param epsilon       minimum percentage of a weight edge to be significant
    * @param maxIterations maximal number of iterations for a non convergent algorithm
    * @param reIndexNodes
    * @param blockSize
    * @return RDD[Assignments] with the vertices added with their respective cluster number
    */
  def run[VD](graph: Graph[VD, Double],
              expansionRate: Int = 2,
              inflationRate: Double = 2.0,
              epsilon: Double = 0.01,
              maxIterations: Int = 10,
              reIndexNodes: Boolean = true,
              blockSize: Int = 1024,
              rePartitionMatrix: Option[Int]
             ): RDD[(VertexId, VertexId)] = {

    new MCL(expansionRate, inflationRate, epsilon, maxIterations).run(graph, reIndexNodes, blockSize, rePartitionMatrix)

  }
}

/** A clustering model for MCL.
  *
  * @see README.md for more details on theory
  * @constructor Constructs an MCL instance with default parameters: {expansionRate: 2, inflationRate: 2, convergenceRate: 0.01, epsilon: 0.05, maxIterations: 10, selfLoopWeight: 0.1, graphOrientationStrategy: "undirected"}.
  * @param expansionRate expansion rate of adjacency matrix at each iteration
  * @param inflationRate inflation rate of adjacency matrix at each iteration
  * @param epsilon       prunning param
  * @param maxIterations maximal number of iterations for a non convergent algorithm
  * @see http://micans.org/mcl/index.html
  */
class MCL private(val expansionRate: Int,
                  val inflationRate: Double,
                  val epsilon: Double,
                  val maxIterations: Int) extends Serializable with Logging {

  def normalization(mat: IndexedRowMatrix): IndexedRowMatrix = {
    new IndexedRowMatrix(
      mat.rows
        .map { row =>
          val svec = row.vector.toSparse
          val sumValues = svec.values.sum
          IndexedRow(row.index,
            new SparseVector(svec.size, svec.indices, svec.values.map(v => v / sumValues)))
        })
  }

  def inflation(mat: IndexedRowMatrix): IndexedRowMatrix = {

    new IndexedRowMatrix(
      mat.rows
        .map { row =>
          val svec = row.vector.toSparse
          IndexedRow(row.index,
            new SparseVector(svec.size, svec.indices, svec.values.map(v => Math.exp(inflationRate * Math.log(v)))))
        }
    )
  }

  def removeWeakConnections(mat: IndexedRowMatrix): IndexedRowMatrix = {
    new IndexedRowMatrix(
      mat.rows.map { row =>
        val svec = row.vector.toSparse
        var numZeros = 0
        var mass = 0.0
        val svecNewValues = svec.values.map(v => {
          if (v < epsilon) {
            numZeros += 1
            0.0
          } else {
            mass += v
            v
          }
        })

        val REC_MIN_MASS = 0.85
        val REC_MIN_NNZ = math.round(1 / epsilon * 0.14).toInt
        //logInfo("REC_MIN_NNZ: " + REC_MIN_NNZ)
        val nnz = svec.indices.size - numZeros

        // in case we pruned too much
        if (nnz < REC_MIN_NNZ && mass < REC_MIN_MASS) {
          val (inds, vals) = (svec.indices zip svec.values).sortBy(-_._2).take(REC_MIN_NNZ).sorted.unzip
          IndexedRow(row.index, new SparseVector(svec.size, inds.toArray, vals.toArray))
        } else
          IndexedRow(row.index, new SparseVector(svec.size, svec.indices, svecNewValues))
      }
    )
  }

  def diff(m1: IndexedRowMatrix, m2: IndexedRowMatrix): Double = {

    val m1RDD: RDD[(Long, SparseVector)] = m1.rows.map((row: IndexedRow) => (row.index, row.vector.toSparse))
    val m2RDD: RDD[(Long, SparseVector)] = m2.rows.map((row: IndexedRow) => (row.index, row.vector.toSparse))

    m1RDD.join(m2RDD).map((tuple: (VertexId, (SparseVector, SparseVector))) => if (tuple._2._1 == tuple._2._2) 0 else 1).sum
  }

  /** Train MCL algorithm.
    *
    * @param graph a graph to partitioned
    * @return an MCLModel where each node is associated to one or more clusters
    */
  def run[VD](graph: Graph[VD, Double], reIndexNodes: Boolean, blockSize: Int, rePartitionMatrix: Option[Int]): RDD[(VertexId,VertexId)] = {

    // Add a new attributes to nodes: a unique row index starting from 0 to transform graph into adjacency matrix

    val lookupTable: RDD[(Int, VertexId)] =
      graph.vertices.zipWithIndex()
        .map(indexedVertex =>
          if (reIndexNodes)
            (indexedVertex._2.toInt, indexedVertex._1._1)
          else
            (indexedVertex._1._1.toInt, indexedVertex._1._1)
        )
        //.toDF("matrixId", "nodeId").cache()

    // replace VD with matrixId
    val preprocessedGraph: Graph[Int, Double] = MCLUtil.preprocessGraph(graph, lookupTable)

    val mat = MCLUtil.toIndexedRowMatrix(preprocessedGraph, rePartitionMatrix)

    // Number of current iterations
    var iter = 0
    // Convergence indicator
    var change = 1.0

    //var M1:IndexedRowMatrix = normalization(mat)
    //var M1:IndexedRowMatrix = new IndexedRowMatrix(normalization(mat).rows.repartition(10000))
    var M1: IndexedRowMatrix = normalization(removeWeakConnections(normalization(mat)))

    while (iter < maxIterations && change > 0) {
      logInfo("================================================  iter " + iter + "  ================================================")

      logInfo("Num of Non-Zeros in Matrix: " + M1.rows.map((row: IndexedRow) => row.vector.toSparse.indices.length ).sum)
      MCLUtil.displayMatrix(M1, format = "%d:%.4f ", predicate = (i) => i == 10)

      val M2: IndexedRowMatrix = normalization(inflation(removeWeakConnections(MCLUtil.expansion(M1, expansionRate, blockSize))))
      M2.rows.cache().checkpoint()

      change = diff(M1, M2)
      logInfo(s"MCL diff is: $change")
      iter = iter + 1
      M1 = M2
    }

    MCLUtil.displayMatrix(M1, format = "%d:%.4f ", predicate = (i) => i == 10)

    // Get attractors in adjacency matrix (nodes with not only null values) and collect every nodes they are attached to in order to form a cluster.
    val rawResult: RDD[(Int, VertexId)] =
      M1.rows.flatMap(r => {
        val sv = r.vector.toSparse
        //sv.indices.map(i => (r.index, i))
        sv.indices.map(i => (i, r.index))
      }) //.toDF("matrixId", "clusterId").cache()

    // Reassign correct ids to each nodes instead of temporary matrix id associated
    val assignmentsRDD: RDD[(VertexId, VertexId)] =
      rawResult
        .join(lookupTable) // , rawResult.col("matrixId") === lookupTable.col("matrixId"))
        //.select($"nodeId", $"clusterId")
        .map(row => (row._2._1, row._2._2))

    assignmentsRDD
  }

}

object MCLUtil {

  val sb = new StringBuilder
  var debugMode = false

  /** Print an adjacency matrix in nice format.
    *
    * @param mat an adjacency matrix
    */
  def displayMatrix(mat: IndexedRowMatrix, format: String = "%d:%.4f ", predicate: Long => Boolean = (_) => true, showFull: Boolean = false): Unit = {
    if (debugMode) {
      println()
      mat
        .rows.sortBy(_.index).collect()
        .foreach(row => {
          if (predicate(row.index)) {
            //printf("%3d => ", row.index)
            printToSB("%3d   ".format(row.index))
            if (showFull)
              row.vector.toArray.foreach(v => printf(format, v))
            else {
              val svec = row.vector.toSparse
              //svec.foreachActive((i, v) => printf("%d:%.3f, ", i, v))
              svec.foreachActive((i, v) => printToSB(format.format(i, v)))
            }
            //println()
            printToSB("\n")
          }
        })
    }

    def printToSB(str: String): Unit = {
      sb.append(str)
      print(str)
    }
  }

  /** Get a suitable graph for MCL model algorithm.
    *
    * Each vertex id in the graph corresponds to a row id in the adjacency matrix.
    *
    * @param graph       original graph
    * @param lookupTable a matching table with nodes ids and new ordered ids
    * @return prepared graph for MCL algorithm
    */
  def preprocessGraph[VD](graph: Graph[VD, Double], lookupTable: RDD[(Int, VertexId)]): Graph[Int, Double] = {
    val newVertices: RDD[(VertexId, Int)] =
      lookupTable.map(
        row => (row._2, row._1)
      )

    Graph(newVertices, graph.edges)
      .groupEdges((e1, e2) => e1 + e2)
  }

  /** Transform a Graph into an IndexedRowMatrix
    *
    * @param graph original graph
    * @return a ready adjacency matrix for MCL process.
    * @todo Check graphOrientationStrategy choice for current graph
    */
  def toIndexedRowMatrix(graph: Graph[Int, Double], rePartitionMatrix: Option[Int] = None) = {

    //Especially relationships values have to be checked before doing what follows
    val rawEntries: RDD[(Int, (Int, Double))] = graph.triplets.map(triplet => (triplet.srcAttr, (triplet.dstAttr, triplet.attr)))

    val numOfNodes: Int = graph.numVertices.toInt

    // add self vertices
    val selfVertices1: VertexRDD[Double] = graph.aggregateMessages[Double](e => e.sendToDst(e.attr), Math.max(_, _), TripletFields.EdgeOnly)
    val selfVertices: RDD[(PartitionID, (PartitionID, Double))] = selfVertices1.join(graph.vertices).map(v => (v._2._2, (v._2._2, v._2._1)))

    val entries: RDD[(Int, (Int, Double))] = rawEntries.union(selfVertices)

    val indexedRows = entries.groupByKey().map(e =>
      IndexedRow(e._1, Vectors.sparse(numOfNodes, e._2.toSeq))
    )

    if (rePartitionMatrix.isEmpty)
      new IndexedRowMatrix(indexedRows, nRows = numOfNodes, nCols = numOfNodes)
    else new IndexedRowMatrix(indexedRows.repartition(rePartitionMatrix.get), nRows = numOfNodes, nCols = numOfNodes)

  }

  def expansion(mat: IndexedRowMatrix, expansionRate: Int = 2, blockSize: Int): IndexedRowMatrix = {
    val bmat = mat.toBlockMatrix(blockSize, blockSize)
    var resmat = bmat
    for (i <- 1 until expansionRate) {
      resmat = BlockMatrixFixedOps.multiply(resmat, bmat)
    }
    resmat.toIndexedRowMatrix()
  }

  /** Transform an IndexedRowMatrix into a Graph
    *
    * @param mat      an adjacency matrix
    * @param vertices vertices of original graph
    * @return associated graph
    */
  def toGraph(mat: IndexedRowMatrix, vertices: RDD[(VertexId, String)]): Graph[String, Double] = {
    val edges: RDD[Edge[Double]] =
      mat.rows.flatMap(f = row => {
        val svec: SparseVector = row.vector.toSparse
        val it: Range = svec.indices.indices
        it.map(ind => Edge(row.index, svec.indices.apply(ind), svec.values.apply(ind)))
      })
    Graph(vertices, edges)
  }
}
