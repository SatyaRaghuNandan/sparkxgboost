package rotationsymmetry.sxgboost

import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.storage.StorageLevel
import rotationsymmetry.sxgboost.loss.Loss


class SparkXGBoostRegressor private[this] (override val uid: String, val loss: Loss)
  extends Predictor[Vector, SparkXGBoostRegressor, SparkXGBoostRegressionModel]
  with SparkXGBoostParams with SparkXGBoostAlgorithm {

  def this(loss: Loss) = this(Identifiable.randomUID("sxgbr"), loss)

  def setAlpha(value: Double): this.type = set(alpha, value)

  def setLambda(value: Double): this.type = set(lambda, value)

  def setEta(value: Double): this.type = set(eta, value)

  def setGamma(value: Double): this.type = set(gamma, value)

  def setNumTrees(value: Int): this.type = set(numTrees, value)

  def setMaxDepth(value: Int): this.type = set(maxDepth, value)

  def setMinInstanceWeight(value: Double): this.type = set(minInstanceWeight, value)

  def setSampleRatio(value: Double): this.type = set(sampleRatio, value)

  def setFeatureSampleRatio(value: Double): this.type = set(featureSampleRatio, value)

  def setMaxConcurrentNodes(value: Int): this.type = set(maxConcurrentNodes, value)

  def setMaxBins(value: Int): this.type = set(maxBins, value)

  def setSeed(value: Long): this.type = set(seed, value)

  override def copy(extra: ParamMap): SparkXGBoostRegressor = {
    val that = new SparkXGBoostRegressor(loss)
    copyValues(that, extra)
  }

  override def train(dataset: DataFrame): SparkXGBoostRegressionModel = {

    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))

    val input: RDD[LabeledPoint] = dataset.select($(labelCol), $(featuresCol)).map {
      case Row(label: Double, features: Vector) => LabeledPoint(label, features)
    }

    input.persist(StorageLevel.MEMORY_AND_DISK)

    val trainedModel = trainModel(input, loss, categoricalFeatures)

    input.unpersist()

    new SparkXGBoostRegressionModel(uid, loss, trainedModel.bias, trainedModel.trees)
  }
}


class SparkXGBoostRegressionModel(
                                   override val uid: String,
                                   val loss: Loss,
                                   val bias: Double,
                                   val trees: List[Node])
  extends PredictionModel[Vector, SparkXGBoostRegressionModel] with SparkXGBoostModelPredictor{

  override def predict(features: Vector): Double = {
    getModelPrediction(features, bias, trees, loss)
  }

  override def copy(extra: ParamMap): SparkXGBoostRegressionModel = {
    val that = new SparkXGBoostRegressionModel(uid, loss, bias, trees)
    copyValues(that, extra).setParent(parent)
  }
}