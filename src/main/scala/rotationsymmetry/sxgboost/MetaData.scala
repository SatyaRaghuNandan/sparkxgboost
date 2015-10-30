package rotationsymmetry.sxgboost

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

private[sxgboost] class MetaData(
      val numFeatures: Int,
      val numBins: Array[Int]) extends Serializable {
}

private[sxgboost] object MetaData {

  def getMetaData(input: RDD[LabeledPoint], splits: Array[Array[Split]]): MetaData = {
    val numFeatures = input.first().features.size
    // The number of Bins is the number of splits + 1
    val numBins = splits.map(_.length + 1)
    new MetaData(numFeatures, numBins)
  }
}