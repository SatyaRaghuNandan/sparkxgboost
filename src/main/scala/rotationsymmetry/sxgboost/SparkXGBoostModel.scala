package rotationsymmetry.sxgboost

import org.apache.spark.mllib.linalg.{Vector, VectorUDT}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}
import rotationsymmetry.sxgboost.loss.Loss

class SparkXGBoostModel(val bias: Double, val trees: List[Node], val loss: Loss) extends Serializable{

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
    val score = if (trees.nonEmpty){
      bias + trees.map{ node => node.predict(features) }.sum
    } else {
      bias
    }
    loss.toPrediction(score)
  }
  def transform(dataset: DataFrame): DataFrame = {
    SchemaUtils.checkColumnType(dataset.schema, featuresCol, new VectorUDT)

    val bcastModel = dataset.sqlContext.sparkContext.broadcast(this)
    val predictUDF = udf { (features: Any) =>
      bcastModel.value.predict(features.asInstanceOf[Vector])
    }
    dataset.withColumn(predictionCol, predictUDF(col(featuresCol)))
  }
}
