package rotationsymmetry.sxgboost

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{udf, col}

class SparkXGBoostModel(val trees: List[Node], val loss: Loss) {

  var featuresCol: String = "features"
  def setFeaturesCol(value: String): this.type = {
    this.featuresCol = value
    this
  }
  var predictionCol: String = "prediction"
  def setPredictionCol(value: String): this.type = {
    this.predictionCol = value
    this
  }

  def predict(features: Vector): Double = {
    val score = trees.map{ node => node.predict(features) }.sum
    loss.toPrediction(score)
  }
  def predict(dataset: DataFrame): DataFrame = {
    val bcastModel = dataset.sqlContext.sparkContext.broadcast(this)
    val predictUDF = udf { (features: Any) =>
      bcastModel.value.predict(features.asInstanceOf[Vector])
    }
    dataset.withColumn(predictionCol, predictUDF(col(featuresCol)))
  }
}
