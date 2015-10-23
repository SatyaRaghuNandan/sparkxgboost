package rotationsymmetry.sxgboost

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

trait TestData {

  val simpleData = Seq(
    LabeledPoint(0.1, Vectors.dense(0, 0)),
    LabeledPoint(0.2, Vectors.dense(0, 1)),
    LabeledPoint(0.3, Vectors.dense(0, 2)),
    LabeledPoint(0.4, Vectors.dense(1, 0)),
    LabeledPoint(0.5, Vectors.dense(1, 1)),
    LabeledPoint(0.6, Vectors.dense(1, 2))
  )

  val simpleBinnedData = Seq(
    TreePoint(0.1, Array(0, 0)),
    TreePoint(0.2, Array(0, 1)),
    TreePoint(0.3, Array(0, 2)),
    TreePoint(0.4, Array(1, 0)),
    TreePoint(0.5, Array(1, 1)),
    TreePoint(0.6, Array(1, 2))
  )

  val simpleMetaData = new MetaData(2, Array(3, 4))

  def randomLabelPointRDD(
      sc: SparkContext,
      numRows: Long,
      numCols: Int,
      numPartitions: Int,
      seed: Long): RDD[LabeledPoint] = {
    val featuresBundle = RandomRDDs.normalVectorRDD(sc, numRows, numCols, numPartitions, seed)
    val labels = RandomRDDs.normalRDD(sc, numRows, numPartitions, seed + 999)

    (labels zip featuresBundle).map { case (label, features) => LabeledPoint(label, features)}
  }

}
