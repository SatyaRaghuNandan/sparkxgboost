package rotationsymmetry.sxgboost

import org.apache.spark.ml.classification.{ProbabilisticClassificationModel, ProbabilisticClassifier}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.storage.StorageLevel
import rotationsymmetry.sxgboost.loss.Loss

class SparkXGBoostClassifier private[this] (override val uid: String, val loss: Loss)
  extends ProbabilisticClassifier[Vector, SparkXGBoostClassifier, SparkXGBoostClassifierModel]
  with SparkXGBoostParams with SparkXGBoostAlgorithm {

  def this(loss: Loss) = this(Identifiable.randomUID("sxgbc"), loss)

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

  set(thresholds -> Array(1.0, 1.0))

  override def copy(extra: ParamMap): SparkXGBoostClassifier = {
    val that = new SparkXGBoostClassifier(loss)
    copyValues(that, extra)
  }

  def train(dataset: DataFrame): SparkXGBoostClassifierModel = {

    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))

    val input: RDD[LabeledPoint] = dataset.select($(labelCol), $(featuresCol)).map {
      case Row(label: Double, features: Vector) => LabeledPoint(label, features)
    }

    input.persist(StorageLevel.MEMORY_AND_DISK)

    val trainedModel = trainModel(input, loss, categoricalFeatures)

    input.unpersist()

    new SparkXGBoostClassifierModel(uid, loss, trainedModel.bias, trainedModel.trees)
  }
}


class SparkXGBoostClassifierModel(
    override val uid: String,
    val loss: Loss,
    val bias: Double,
    val trees: List[Node])
  extends ProbabilisticClassificationModel[Vector, SparkXGBoostClassifierModel] with SparkXGBoostModelPredictor{

  val numClasses = 2

  def predictRaw(features: Vector): Vector = {
    val p1 = getModelPrediction(features, bias, trees, loss)
    Vectors.dense(1 - p1, p1)
  }

  def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction
  }

  override def copy(extra: ParamMap): SparkXGBoostClassifierModel = {
    val that = new SparkXGBoostClassifierModel(uid, loss, bias, trees)
    copyValues(that, extra).setParent(parent)
  }
}