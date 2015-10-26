package rotationsymmetry.sxgboost

import org.apache.spark.mllib.linalg.{Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.storage.StorageLevel
import rotationsymmetry.sxgboost.loss.Loss

import scala.collection.mutable
import scala.util.Random


class SparkXGBoost(val loss: Loss) {
  var numTrees : Int = 1
  def setNumTrees(value: Int): this.type = {
    require(value >= 1)
    this.numTrees = value
    this
  }

  var lambda: Double = 0.0
  def setLambda(value: Double): this.type = {
    require(value >= 0.0)
    this.lambda = value
    this
  }

  var alpha: Double = 0.0
  def setAlpha(value: Double): this.type = {
    require(value >= 0.0)
    this.alpha = value
    this
  }

  var gamma: Double = 0.0
  def setGamma(value: Double): this.type = {
    require(value >= 0.0)
    this.gamma = value
    this
  }

  var labelCol: String ="label"
  def setLabelCol(value: String): this.type = {
    this.labelCol = value
    this
  }

  var featuresCol: String ="features"
  def setFeaturesCol(value: String): this.type = {
    this.featuresCol = value
    this
  }

  var maxBins: Int = 32
  def setMaxBins(value: Int): this.type = {
    require(value >= 2)
    this.maxBins = value
    this
  }

  var maxDepth: Int = 5
  def setMaxDepth(value: Int): this.type = {
    require(value >= 1)
    this.maxDepth = value
    this
  }

  var minWeight: Double = 1.0
  def setMinWeight(value: Double): this.type = {
    require(value >= 1.0)
    this.minWeight = value
    this
  }

  var featureSampleRatio: Double = 1.0
  def setFeatureSampleRatio(value: Double): this.type = {
    require(value > 0 && value <= 1.0)
    this.featureSampleRatio = value
    this
  }
  
  var sampleRatio: Double = 1.0
  def setSampleRatio(value: Double): this.type = {
    require(value > 0 && value <= 1.0)
    this.sampleRatio = value
    this
  }

  var maxConcurrentNodes: Int = 50
  def setMaxConcurrentNodes(value: Int): this.type = {
    require(value >= 1)
    this.maxConcurrentNodes = value
    this
  }

  def fit(dataset: DataFrame): SparkXGBoostModel = {

    // Check dataset schema
    //SchemaUtils.checkColumnType(dataset.schema, featuresCol, new VectorUDT)
    SchemaUtils.checkColumnType(dataset.schema, labelCol, DoubleType)

    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema(featuresCol))

    val input: RDD[LabeledPoint] = dataset.select(labelCol, featuresCol).map {
      case Row(label: Double, features: Vector) => LabeledPoint(label, features)
    }
    input.persist(StorageLevel.MEMORY_AND_DISK)

    val splits = OrderedSplit.createOrderedSplits(input, categoricalFeatures, maxBins)

    val metaData = MetaData.getMetaData(input, splits)
    val treePoints = TreePoint.convertToTreeRDD(input, splits)

    val bias = loss.getInitialBias(input)
    val workingModel = new WorkingModel(bias, Array())

    var treeIdx: Int = 0
    while (treeIdx < numTrees){

      val currentRoot = new WorkingNode(0)
      val nodeQueue: mutable.Queue[WorkingNode] = new mutable.Queue[WorkingNode]()
      nodeQueue.enqueue(currentRoot)

      while (nodeQueue.nonEmpty){
        val nodeBatch = dequeueWithinMemLimit(nodeQueue)
        val featureIndicesBundle = sampleFeatureIndices(metaData.numFeatures, featureSampleRatio, nodeBatch.length)

        val sampledTreePoints = treePoints.sample(false, sampleRatio)
        // TODO: Broadcast?
        val lossAggregator = sampledTreePoints.treeAggregate(
          new LossAggregator(featureIndicesBundle, workingModel, currentRoot, metaData, loss))(
          seqOp = (agg, treePoint) => agg.add(treePoint),
          combOp = (agg1, agg2) => agg1.merge(agg2))

        val nodesToEnqueue = nodeBatch.indices.flatMap { nodeIdx =>
          nodeBatch(nodeIdx).idxInBatch = None

          val bestSplit = findBestSplit(lossAggregator.stats(nodeIdx),
            lossAggregator.featureIndicesBundle(nodeIdx), lossAggregator.offsets(nodeIdx), lambda, alpha, gamma)

          bestSplit match {
            case Some(splitInfo) => {
              val node = nodeBatch(nodeIdx)
              node.split = Some(splitInfo.split)

              val leftChild = new WorkingNode(node.depth + 1)
              leftChild.prediction = Some(splitInfo.leftPrediction)
              leftChild.weight = Some(splitInfo.leftWeight)
              node.leftChild = Some(leftChild)

              val rightChild = new WorkingNode(node.depth + 1)
              rightChild.prediction = Some(splitInfo.rightPrediction)
              rightChild.weight = Some(splitInfo.rightWeight)
              node.rightChild = Some(rightChild)

              Iterator(leftChild, rightChild)
                .filter(workingNode => workingNode.depth < maxDepth && workingNode.weight.get >= minWeight)
            }
            case None => Iterator()
          }
        }
        nodeQueue ++= nodesToEnqueue
      }

      if (!currentRoot.isLeaf) {
        workingModel.trees = workingModel.trees :+ currentRoot
        treeIdx += 1
      } else {
        treeIdx = numTrees // breaking the loop
      }
    }

    input.unpersist()

    workingModel.toSparkXGBoostModel(splits, loss).setFeaturesCol(featuresCol)
  }

  private def dequeueWithinMemLimit(queue: mutable.Queue[WorkingNode]): Array[WorkingNode] = {
    val arrayBuilder = mutable.ArrayBuilder.make[WorkingNode]()

    var idx: Int = 0
    while (queue.nonEmpty && idx < maxConcurrentNodes){
      val node = queue.dequeue()
      node.idxInBatch = Some(idx)
      arrayBuilder += node
      idx += 1
    }
    arrayBuilder.result()
  }

  private[sxgboost] def extractDiffsAndWeightsFromStatsView(v: Seq[Double]): (Array[Double], Array[Double], Array[Double]) = {
    val length = v.length / 3
    val d1 = new Array[Double](length)
    val d2 = new Array[Double](length)
    val weights = new Array[Double](length)
    var idx = 0
    while (idx < length){
      d1(idx) = v(idx * 3)
      d2(idx) = v(idx * 3 + 1)
      weights(idx) = v(idx * 3 + 2)
      idx += 1
    }
    (d1, d2, weights)
  }

  private[sxgboost] def findBestSplitForSingleFeature(
   statsView: Seq[Double],
   featureIdx: Int,
   lambda: Double,
   alpha: Double) = {
    val (d1, d2, weights) = extractDiffsAndWeightsFromStatsView(statsView)
    val (d1CuSum, d1Total) = getCuSumAndTotal(d1)
    val (d2CuSum, d2Total) = getCuSumAndTotal(d2)
    val (weightsCuSum, weightsTotal) = getCuSumAndTotal(weights)

    val partialObjAndEstForParent = getPartialObjAndEst(d1Total, d2Total, lambda, alpha)
    val partialObjAndEst = (d1CuSum zip d2CuSum).map { case (d1Left, d2Left) =>
      val left = getPartialObjAndEst(d1Left, d2Left, lambda, alpha)
      val right = getPartialObjAndEst(d1Total - d1Left, d2Total - d2Left, lambda, alpha)
      (left._1 + right._1 , left._2, right._2)
    }
    val optimIdx = partialObjAndEst.zipWithIndex.min._2

    val gain = partialObjAndEstForParent._1 - partialObjAndEst(optimIdx)._1
    val leftPrediction = partialObjAndEst(optimIdx)._2
    val rightPrediction = partialObjAndEst(optimIdx)._3
    val leftWeight = weightsCuSum(optimIdx)
    val rightWeight = weightsTotal - leftWeight

    SplitInfo(WorkingSplit(featureIdx, optimIdx), gain,
      leftPrediction, rightPrediction, leftWeight, rightWeight)
  }

  private def createStatsViews(stats: Array[Double], featureIndices: Array[Int], offsets: Array[Int]) = {
    val extOffsets = offsets :+ stats.length

    featureIndices.indices.map { idx=>
      stats.view(extOffsets(idx), extOffsets(idx + 1))
    }
  }

  private[sxgboost] def findBestSplit(
       stats: Array[Double],
       featureIndices: Array[Int],
       offsets: Array[Int],
       lambda: Double,
       alpha: Double,
       gamma: Double): Option[SplitInfo] = {
    val statsViews = createStatsViews(stats, featureIndices, offsets)

    val candidateSplit = (statsViews zip featureIndices) map { case (statsView, featureIdx)=>
        findBestSplitForSingleFeature(statsView, featureIdx, lambda, alpha)
    }

    val eligibleSplit = candidateSplit.filter(_.gain > gamma)

    if (eligibleSplit.nonEmpty){
      Some(eligibleSplit.maxBy(_.gain))
    } else {
      None
    }
  }

  private[sxgboost] def getCuSumAndTotal(ds: Seq[Double]) = {
    val cusum: Seq[Double] = ds.scan(0.0)(_+_)
    (cusum.drop(1).dropRight(1), cusum.last)
  }


  private[sxgboost] def sampleFeatureIndices(numFeatures: Int, featureSampleRatio: Double, numSamples: Int): Array[Array[Int]] = {
    val numSampledFeatures = Math.ceil(numFeatures * featureSampleRatio).toInt
    val indices = Range(0, numFeatures).toBuffer
    val rnd = new Random()
    val arrayBuilder = mutable.ArrayBuilder.make[Array[Int]]()
    Range(0, numSamples).foreach { i =>
      arrayBuilder += rnd.shuffle(indices).take(numSampledFeatures).toArray
    }
    arrayBuilder.result()
  }

  private def getNonZero(value: Double) = if (Math.abs(value) > 1e-10) value else 1e-10

  private[sxgboost] def getPartialObjAndEst(g: Double, h: Double, lambda: Double, alpha: Double): (Double, Double) = {
    val denom = getNonZero(h + lambda)

    val estPositiveSide = if (-(g + alpha) / denom >= 0) {
      - (g + alpha) / denom
    } else {
      0.0
    }
    val objPositiveSide = calculatePartialObj(g, h, lambda, alpha, estPositiveSide)

    val estNegativeSide = if (-(g - alpha) / denom < 0) {
      - (g - alpha) / denom
    } else {
      0.0
    }
    val objNegativeSide = calculatePartialObj(g, h, lambda, alpha, estNegativeSide)

    if (objPositiveSide < objNegativeSide) {
      (objPositiveSide, estPositiveSide)
    } else {
      (objNegativeSide, estNegativeSide)
    }
  }

  private def calculatePartialObj(g: Double, h: Double, lambda: Double, alpha: Double, est: Double): Double = {
    g * est + 0.5 * (h + lambda) * Math.pow(est, 2.0) + alpha * Math.abs(est)
  }

}
