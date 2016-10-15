package org.apache.spark.mllib.linalg.distributed

import org.apache.spark.mllib.linalg.{Matrix, SparseMatrix, SparseMatrixOps}
import org.apache.spark.rdd.RDD
import org.apache.spark.{Logging, SparkException}

import scala.collection.mutable

/**
  * Created by oraviv on 15/10/2016.
  */
object BlockMatrixFixedOps extends Logging {

  /** Block (i,j) --> Set of destination partitions */
  private type BlockDestinations = Map[(Int, Int), Set[Int]]

  private[distributed] def simulateMultiply( me: BlockMatrix,
                                             other: BlockMatrix,
                                             partitioner: GridPartitioner): (BlockDestinations, BlockDestinations) = {
    val leftMatrix = me.blocks.keys.collect() // blockInfo should already be cached
    val rightMatrix = other.blocks.keys.collect()

    val rightCounterpartsHelper = rightMatrix.groupBy(_._1).mapValues(_.map(_._2))
    val leftDestinations = leftMatrix.map { case (rowIndex, colIndex) =>
      val rightCounterparts = rightCounterpartsHelper.getOrElse(colIndex, Array())
      val partitions = rightCounterparts.map(b => partitioner.getPartition((rowIndex, b)))
      ((rowIndex, colIndex), partitions.toSet)
    }.toMap

    val leftCounterpartsHelper = leftMatrix.groupBy(_._2).mapValues(_.map(_._1))
    val rightDestinations = rightMatrix.map { case (rowIndex, colIndex) =>
      val leftCounterparts = leftCounterpartsHelper.getOrElse(rowIndex, Array())
      val partitions = leftCounterparts.map(b => partitioner.getPartition((b, colIndex)))
      ((rowIndex, colIndex), partitions.toSet)
    }.toMap

    (leftDestinations, rightDestinations)
  }


  def multiply(me:BlockMatrix, other: BlockMatrix): BlockMatrix = {
    require(me.numCols() == other.numRows(), "The number of columns of A and the number of rows " +
      s"of B must be equal. A.numCols: ${me.numCols()}, B.numRows: ${other.numRows()}. If you " +
      "think they should be equal, try setting the dimensions of A and B explicitly while " +
      "initializing them.")

    logInfo(s"Multiplying big matrices: (${me.numRows()}x${me.numCols()}) with (${other.numRows()}x${other.numCols()})")
    logInfo(s"Blocks (${me.numRowBlocks}x${me.numColBlocks}) with (${other.numRowBlocks}x${other.numColBlocks})")
    logInfo(s"perBlock (${me.rowsPerBlock}x${me.colsPerBlock}) with (${other.rowsPerBlock}x${other.colsPerBlock})")


    var addingCounter = 0

    if (me.colsPerBlock == other.rowsPerBlock) {
      val resultPartitioner = GridPartitioner(me.numRowBlocks, other.numColBlocks,
        math.max(me.blocks.partitions.length, other.blocks.partitions.length))
        //me.numRowBlocks*other.numColBlocks/4)
      val (leftDestinations, rightDestinations) = simulateMultiply(me, other, resultPartitioner)
      // Each block of A must be multiplied with the corresponding blocks in the columns of B.
      val flatA = me.blocks.flatMap { case ((blockRowIndex, blockColIndex), block) =>
        val destinations = leftDestinations.getOrElse((blockRowIndex, blockColIndex), Set.empty)
        destinations.map(j => (j, (blockRowIndex, blockColIndex, block)))
      }
      // Each block of B must be multiplied with the corresponding blocks in each row of A.
      val flatB = other.blocks.flatMap { case ((blockRowIndex, blockColIndex), block) =>
        val destinations = rightDestinations.getOrElse((blockRowIndex, blockColIndex), Set.empty)
        destinations.map(j => (j, (blockRowIndex, blockColIndex, block)))
      }
      val newBlocks = flatA.cogroup(flatB, resultPartitioner).flatMap { case (pId, (a, b)) =>

        logInfo(s"Partition $pId/${resultPartitioner.numPartitions}, block-list sizes: ${a.size}x${b.size}")
        val matrixHelper = b.groupBy(_._1)
        val hits = a.map(t => matrixHelper.getOrElse(t._1, List()).size).sum
        logInfo(s"hits (real multiplications): $hits")
        logInfo(s"hit rate in block-lists: ${1.0 * hits / a.size / b.size * 100}%")

        val targetBlockHits2 = a.foldLeft(mutable.HashSet.empty[(Int, Int)]) { case (set, (i, j, m)) => set ++= matrixHelper.getOrElse(i, List()).map(bjk => (i, bjk._2)) }.size
        logInfo("# of hit blocks in partition: " + targetBlockHits2)

        logInfo(s"block hit rate in partition: ${1.0 * targetBlockHits2 / (me.numColBlocks * me.numRowBlocks / resultPartitioner.numPartitions) * 100}%")
        var hitCount = 0

        a.flatMap { case (leftRowIndex, leftColIndex, leftBlock) =>
          //b.filter(_._1 == leftColIndex).map { case (rightRowIndex, rightColIndex, rightBlock) =>
          matrixHelper.getOrElse(leftColIndex, List()).map { case (rightRowIndex, rightColIndex, rightBlock) =>

            if (hitCount % math.max(1, hits / 10) == 0) logInfo(s"calculated $hitCount multiplications (${hitCount * 100 / hits}%)")
            hitCount += 1

            val C = rightBlock match {
              //case dense: DenseMatrix => leftBlock.multiply(dense)
              case sparse: SparseMatrix => {
                //logInfo("blocks multiplying: "+leftBlock.numRows+"x"+leftBlock.numCols+" X "+sparse.numRows+"x"+sparse.numCols);
                //val r = leftBlock.multiply(sparse.toDense).toSparse
                val r = SparseMatrixOps.outerMultiply(leftBlock.asInstanceOf[SparseMatrix], sparse)
                r
              }
              //case sparse: SparseMatrix => leftBlock.multiply(sparse.toDense)
              case _ =>
                throw new SparkException(s"Unrecognized matrix type ${rightBlock.getClass}.")
            }

            //((leftRowIndex, rightColIndex), C.toBreeze)
            ((leftRowIndex, rightColIndex), C)
          }
        }
        //}.reduceByKey(resultPartitioner, (a, b) => {val c = a + b; c.asInstanceOf[CSCMatrix[Double]].compact(); c}).mapValues(m => {
        //}.reduceByKey(resultPartitioner, (a, b) => {}) //.mapValues(m => {
      }.aggregateByKey((0,0,new mutable.HashMap[(Int, Int), Double]()), resultPartitioner)((agg,sm) => (sm.numRows, sm.numCols, SparseMatrixOps.sparseAdd(agg._3,sm)), (a,b) => (a._1,a._2,{a._3 ++= b._3; a._3}))
       .mapValues{case (r,c,lst) => SparseMatrix.fromCOO(r, c, lst.map { case ((i, j), d) => (i, j, d) })}
//      }.groupBy(_._1).map{ case ((a,b), l ) => {
//                                    if (addingCounter%1==0)
//                                      logInfo(s"adding $addingCounter");
//                                    addingCounter+=1;
//                                    ((a,b), l.map(_._2).reduce((a, b) => {val c = a + b; c.asInstanceOf[CSCMatrix[Double]].compact(); c}))
//      }}.map{case ((a,b), m) => {
                                          //val bdm = m.toDenseMatrix
                                          //val dm = new DenseMatrix(bdm.rows, bdm.cols, bdm.data, bdm.isTranspose)
                                          //dm.toSparse
                                          //dm
        //((a,b), Matrices.fromBreeze(m))
        //                            })

      new BlockMatrix(newBlocks.asInstanceOf[RDD[((Int, Int), Matrix)]], me.rowsPerBlock, other.colsPerBlock, me.numRows(), other.numCols())
    } else {
      throw new SparkException("colsPerBlock of A doesn't match rowsPerBlock of B. " +
        s"A.colsPerBlock: $me.colsPerBlock, B.rowsPerBlock: ${other.rowsPerBlock}")
    }
  }
}
