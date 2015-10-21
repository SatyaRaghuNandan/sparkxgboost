package rotationsymmetry.sxgboost

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, DataFrame}

import scala.collection.mutable
import scala.util.Random


class SparkXGBoost {
  var numTrees : Int = 1
  def setNumTrees(value: Int): this.type = {
    this.numTrees = value
    this
  }

  var loss: Loss = null
  def setLoss(value: Loss): this.type = {
    this.loss = value
    this
  }

  var lambda: Double = 0
  def setLambda(value: Double): this.type = {
    this.lambda = value
    this
  }

  var gamma: Double = 0
  def setGamma(value: Double): this.type = {
    this.gamma = value
    this
  }

  var labelCol: String ="label"
  def setInputCol(value: String): this.type = {
    this.labelCol = value
    this
  }

  var featuresCol: String ="features"
  def setFeaturesCol(value: String): this.type = {
    this.featuresCol = value
    this
  }

  var maxBins: Int = 100
  def setMaxBins(value: Int): this.type = {
    this.maxBins = value
    this
  }

  var maxDepth: Int = 1
  def setMaxDepth(value: Int): this.type = {
    this.maxDepth = value
    this
  }

  var minWeight: Double = 1.0
  def setMinWeight(value: Double): this.type = {
    this.minWeight = value
    this
  }


  def train(dataset: DataFrame): SparkXGBoostModel = {

    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema(featuresCol))

    val input: RDD[LabeledPoint] = dataset.select(labelCol, featuresCol).map {
      case Row(label: Double, features: Vector) => LabeledPoint(label, features)
    }

    val splits = OrderedSplit.createOrderedSplits(input, categoricalFeatures, maxBins)

    val metaData = MetaData.getMetaData(input, splits)
    val treePoints = TreePoint.convertToTreeRDD(input, splits)

    val initialBias = loss.getInitialBias(input)
    val initialTree = new WorkingNode(0)
    initialTree.prediction = Some(initialBias)
    var workingModel = WorkingModel(Array(initialTree))

    var treeIdx: Int = 2
    while (treeIdx < numTrees){

      val currentRoot = new WorkingNode(0)
      val nodeQueue: mutable.Queue[WorkingNode] = new mutable.Queue[WorkingNode]()
      nodeQueue.enqueue(currentRoot)

      while (nodeQueue.nonEmpty){
        val nodeBatch = dequeueWithinMemLimit(nodeQueue)
        val featureIndicesBundle = sampleFeatureIndices(metaData.numFeatures, 10, nodeBatch.length)

        val lossAggregator = treePoints.treeAggregate(
          new LossAggregator(featureIndicesBundle, workingModel, currentRoot, metaData, loss))(
          seqOp = (agg, treePoint) => agg.add(treePoint),
          combOp = (agg1, agg2) => agg1.merge(agg2))

        val nodesToEnqueue = nodeBatch.indices.flatMap { nodeIdx =>
          val bestSplit = findBestSplit(lossAggregator.stats(nodeIdx),
            lossAggregator.featureIndicesBundle(nodeIdx), lossAggregator.offsets(nodeIdx), lambda, gamma)

          bestSplit match {
            case Some(splitInfo) => {
              val node = nodeBatch(nodeIdx)
              node.split = Some(splitInfo.split)
              node.idxInBatch = None

              val leftChild = new WorkingNode(node.depth + 1)
              leftChild.prediction = Some(splitInfo.leftPrediction)
              leftChild.weight = Some(splitInfo.leftWeight)
              node.leftChild = Some(leftChild)

              val rightChild = new WorkingNode(node.depth + 1)
              rightChild.prediction = Some(splitInfo.rightPrediction)
              rightChild.weight = Some(splitInfo.rightWeight)
              node.rightChild = Some(rightChild)

              Iterator(leftChild, rightChild)
                .filter(workingNode => workingNode.depth <= maxDepth && workingNode.weight.get >= minWeight)
            }
            case None => Iterator()
          }
        }
        nodeQueue ++= nodesToEnqueue
      }

      if (!currentRoot.isLeaf) {
        workingModel = WorkingModel(workingModel.trees :+ currentRoot)
        treeIdx += 1
      } else {
        treeIdx = numTrees // breaking the loop
      }
    }

    workingModel.toSparkXGBoostModel(splits, loss)
  }

  def dequeueWithinMemLimit(queue: mutable.Queue[WorkingNode]): Array[WorkingNode] = {
    val arrayBuilder = mutable.ArrayBuilder.make[WorkingNode]()

    var idx: Int = 0
    while (queue.nonEmpty && idx < 5){
      val node = queue.dequeue()
      node.idxInBatch = Some(idx)
      arrayBuilder += node
      idx += 1
    }
    arrayBuilder.result()
  }

  def extractDiffsAndWeightsFromStatsView(v: Seq[Double]): (Array[Double], Array[Double], Array[Double]) = {
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

  def findBestSplitForSingleFeature(
   statsView: Seq[Double],
   featureIdx: Int,
   lambda: Double) = {
    val (d1, d2, weights) = extractDiffsAndWeightsFromStatsView(statsView)
    val (d1CuSum, d1Total) = getCuSumAndTotal(d1)
    val (d2CuSum, d2Total) = getCuSumAndTotal(d2)
    val (weightsCuSum, weightsTotal) = getCuSumAndTotal(weights)
    val parentTerm = getObjRatio(d1Total, d2Total, lambda)
    val gains = (d1CuSum zip d2CuSum).map { case (d1Left, d2Left) =>
        val leftTerm = getObjRatio(d1Left, d2Left, lambda)
        val rightTerm = getObjRatio(d1Total - d1Left, d2Total - d2Left, lambda)
        0.5 * (leftTerm + rightTerm - parentTerm)
    }

    val optimIdx = gains.zipWithIndex.max._2
    val leftPrediction = getPrediction(d1CuSum(optimIdx), d2CuSum(optimIdx), lambda)
    val rightPrediction = getPrediction(
      d1Total - d1CuSum(optimIdx), d2Total - d2CuSum(optimIdx), lambda)

    val leftWeight = weightsCuSum(optimIdx)
    val rightWeight = weightsTotal - leftWeight
    SplitInfo(WorkingSplit(featureIdx, optimIdx), gains(optimIdx),
      leftPrediction, rightPrediction, leftWeight, rightWeight)
  }

  def createStatsViews(stats: Array[Double], featureIndices: Array[Int], offsets: Array[Int]) = {
    val extOffsets = offsets :+ stats.length

    featureIndices.indices.map { idx=>
      stats.view(extOffsets(idx), extOffsets(idx + 1))
    }
  }

  def findBestSplit(
       stats: Array[Double],
       featureIndices: Array[Int],
       offsets: Array[Int],
       lambda: Double,
       gamma: Double): Option[SplitInfo] = {
    val statsViews = createStatsViews(stats, featureIndices, offsets)

    val candidateSplit = (statsViews zip featureIndices) map { case (statsView, featureIdx)=>
        findBestSplitForSingleFeature(statsView, featureIdx, lambda)
    }

    val eligibleSplit = candidateSplit.filter(_.gain > gamma)

    if (eligibleSplit.nonEmpty){
      Some(eligibleSplit.maxBy(_.gain))
    } else {
      None
    }
  }

  def getCuSumAndTotal(ds: Seq[Double]) = {
    val cusum: Seq[Double] = ds.scan(0.0)(_+_)
    (cusum.drop(1).dropRight(1), cusum.last)
  }

  def getObjRatio(g: Double, h: Double, lambda: Double) = {
    (g * g) / getNonZero(h + lambda)
  }

  def getPrediction(g: Double, h: Double, lambda: Double) = {
    - g / getNonZero(h + lambda)
  }

  def getNonZero(value: Double) = if (Math.abs(value) > 1e-10) value else 1e-10

  def sampleFeatureIndices(numFeatures: Int, numSampledFeatures: Int, numSamples: Int): Array[Array[Int]] = {
    val indices = Range(0, numFeatures).toBuffer
    val rnd = new Random()
    val arrayBuilder = mutable.ArrayBuilder.make[Array[Int]]()
    Range(0, numSamples).foreach { i =>
      arrayBuilder += rnd.shuffle(indices).take(numSampledFeatures).toArray
    }
    arrayBuilder.result()
  }

}
