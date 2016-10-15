package org.apache.spark.mllib.linalg

import breeze.linalg.CSCMatrix
import org.apache.spark.Logging

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
  * Created by oraviv on 16/04/2016.
  */
object SparseMatrixOps extends Logging {

  def myToBreeze(s: SparseMatrix): CSCMatrix[Double] = {
    if (!s.isTransposed) {
      new CSCMatrix[Double](s.values, s.numRows, s.numCols, s.colPtrs, s.rowIndices)
    } else {
      val breezeMatrix = new CSCMatrix[Double](s.values, s.numCols, s.numRows, s.colPtrs, s.rowIndices)
      breezeMatrix.t
    }
  }

  def myFromBreeze(csc: CSCMatrix[Double]): SparseMatrix = new SparseMatrix(csc.rows, csc.cols, csc.colPtrs, csc.rowIndices, csc.data)

  // (oraviv) I took the original gemm implementation from breeze, stripped it to the only case we need and added Bval != 0 check.
  def gemmPlus(A: SparseMatrix, Bs: SparseMatrix): DenseMatrix = {

    val B = Bs.toDense
    val C = DenseMatrix.zeros(A.numRows, B.numCols);


    val mA: Int = A.numRows
    val nB: Int = B.numCols
    val kA: Int = A.numCols
    val kB: Int = B.numRows

    require(kA == kB, s"The columns of A don't match the rows of B. A: $kA, B: $kB")
    require(mA == C.numRows, s"The rows of C don't match the rows of A. C: ${C.numRows}, A: $mA")
    require(nB == C.numCols,
      s"The columns of C don't match the columns of B. C: ${C.numCols}, A: $nB")

    val Avals = A.values
    val Bvals = B.values
    val Cvals = C.values
    val ArowIndices = A.rowIndices
    val AcolPtrs = A.colPtrs

    var counter = 0

    // Perform matrix multiplication and add to C. The rows of A are multiplied by the columns of
    // B, and added to C.
    var colCounterForB = 0 // the column to be updated in C
    // Expensive to put the check inside the loop
    while (colCounterForB < nB) {
      var colCounterForA = 0 // The column of A to multiply with the row of B
      val Bstart = colCounterForB * kB
      val Cstart = colCounterForB * mA
      while (colCounterForA < kA) {
        var i = AcolPtrs(colCounterForA)
        val indEnd = AcolPtrs(colCounterForA + 1)
        val Bval = Bvals(Bstart + colCounterForA)
        if (Bval != 0)
          while (i < indEnd) {
            Cvals(Cstart + ArowIndices(i)) += Avals(i) * Bval
            counter += 1
            i += 1
          }
        colCounterForA += 1
      }
      colCounterForB += 1
    }

    println("asd1: " + counter)

    C
  }

  // (oraviv) my take-off of the Sparse-Sparse Gemm implementation - surprisingly doesn't run much faster than @gemmPlus.
  // just doesn't run into OutOfMemory with big matrices.
  def outerMultiply(A: SparseMatrix, Bt: SparseMatrix): SparseMatrix = {
    // it would be much faster for native MLlib
    val B = myToBreeze(Bt).t

    val Clist = new mutable.HashMap[(Int, Int), Double]()

    var counter = 0

    var k = 0
    while (k < A.numCols) {
      var iB = B.colPtrs(k)
      val iBEnd = B.colPtrs(k + 1)
      while (iB < iBEnd) {
        //val Bcol = BrowIndices(iB)*numCols
        val Bcol = B.rowIndices(iB)
        var iA = A.colPtrs(k)
        val iAEnd = A.colPtrs(k + 1)
        val Bval = B.data(iB)
        while (iA < iAEnd) {
          //Cvals(Bcol + ArowIndices(iA)) += Avals(iA) * Bval
          Clist.put((A.rowIndices(iA), Bcol), A.values(iA) * Bval + Clist.getOrElse((A.rowIndices(iA), Bcol), 0.0))
          counter += 1
          iA += 1
        }
        iB += 1
      }
      k += 1
    }

    //logInfo("num calculations: " + counter)

    SparseMatrix.fromCOO(A.numRows, Bt.numCols, Clist.map { case ((i, j), d) => (i, j, d) })
  }

  def sparseAdd(Clist: mutable.HashMap[(Int, Int), Double], sm: SparseMatrix): mutable.HashMap[(Int, Int), Double] = {

    def addToC(s: SparseMatrix) = {
      var c = 0
      while (c < s.numCols) {
        var r = s.colPtrs(c)
        val rEnd = s.colPtrs(c + 1)
        while (r < rEnd) {
          Clist.put((s.rowIndices(r), c), s.values(r) + Clist.getOrElse((s.rowIndices(r), c), 0.0))
          r += 1
        }
        c += 1
      }
    }

    addToC(sm)

    //SparseMatrix.fromCOO(A.numRows, A.numCols, Clist.map { case ((i, j), d) => (i, j, d) })
    Clist
  }

  def sparseAdd(Clist: Array[mutable.HashMap[Int, Double]], sm: SparseMatrix): Array[mutable.HashMap[Int, Double]] = {

    var Clist2 = Clist
    if (Clist.size==0)
      Clist2 = new Array[mutable.HashMap[Int, Double]](sm.numCols)

    def addToC(s: SparseMatrix) = {
      var c = 0
      while (c < s.numCols) {
        var r = s.colPtrs(c)
        val rEnd = s.colPtrs(c + 1)
        while (r < rEnd) {
          if (Clist2(c)==null)
            Clist2(c) = new mutable.HashMap[Int, Double]()

          Clist2(c).put(s.rowIndices(r), s.values(r) + Clist2(c).getOrElse(s.rowIndices(r), 0.0))
          r += 1
        }
        c += 1
      }
    }

    addToC(sm)

    //SparseMatrix.fromCOO(A.numRows, A.numCols, Clist.map { case ((i, j), d) => (i, j, d) })
    Clist2
  }

  def sparseAdd2(d: DenseMatrix, s: SparseMatrix): DenseMatrix = {

    var c = 0
    while (c < s.numCols) {
      var r = s.colPtrs(c)
      val rEnd = s.colPtrs(c + 1)
      while (r < rEnd) {
        //Clist.put((s.rowIndices(r), c), s.values(r) + Clist.getOrElse((s.rowIndices(r), c), 0.0))
        //d(r,c) += s.values(r)
        d.values(c*d.numRows + s.rowIndices(r)) += s.values(r)
        r += 1
      }
      c += 1
    }

    d

  }

  def sparseAdd(s1: SparseVector, s2: SparseVector): SparseVector = {
    val indices = ArrayBuffer[Int]()
    val values = ArrayBuffer[Double]()

    var i1 = 0
    var i2 = 0
    while (i1 < s1.indices.length && i2 < s2.indices.length) {
      if (s1.indices(i1)==s2.indices(i2)) {
        indices += s1.indices(i1)
        values += s1.values(i1) + s2.values(i2)
        i1+=1
        i2+=1
      } else if (s1.indices(i1)<s2.indices(i2)) {
        indices += s1.indices(i1)
        values += s1.values(i1)
        i1+=1
      } else {
        indices += s2.indices(i2)
        values += s2.values(i2)
        i2+=1
      }
    }

    while (i1 < s1.indices.length) {
      indices += s1.indices(i1)
      values += s1.values(i1)
      i1+=1
    }

    while (i2 < s2.indices.length) {
      indices += s2.indices(i2)
      values += s2.values(i2)
      i2+=1
    }

    new SparseVector(s1.size, indices.toArray, values.toArray)

  }

  def sparseMinus(s1: SparseVector, s2: SparseVector): SparseVector = {
    val s2m = new SparseVector(s2.size, s2.indices, s2.values.map(-_))
    sparseAdd(s1, s2m)
  }

}
