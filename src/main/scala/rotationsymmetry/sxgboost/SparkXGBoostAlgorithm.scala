package rotationsymmetry.sxgboost

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import rotationsymmetry.sxgboost.loss.Loss

import scala.collection.mutable
import scala.util.Random

trait SparkXGBoostAlgorithm {
  self: SparkXGBoostParams =>

  def trainModel(input: RDD[LabeledPoint], loss: Loss, categoricalFeatures: Map[Int, Int]): TrainedModel = {

    val rng = new Random($(seed))

    val splits = OrderedSplit.createOrderedSplits(input, categoricalFeatures, $(maxBins), rng.nextLong())

    val metaData = MetaData.getMetaData(input, splits)
    val treePoints = TreePoint.convertToTreeRDD(input, splits)

    val bias = loss.getInitialBias(input)
    val workingModel = new WorkingModel(bias, Array())

    var treeIdx: Int = 0
    while (treeIdx < $(numTrees)){

      val currentRoot = new WorkingNode(0)
      val nodeQueue: mutable.Queue[WorkingNode] = new mutable.Queue[WorkingNode]()
      nodeQueue.enqueue(currentRoot)

      while (nodeQueue.nonEmpty){
        val nodeBatch = dequeueWithinMemLimit(nodeQueue)
        val featureIndicesBundle = sampleFeatureIndices(metaData.numFeatures, $(featureSampleRatio), nodeBatch.length, rng)

        val sampledTreePoints = treePoints.sample(withReplacement = false, $(sampleRatio), rng.nextLong())
        // TODO: Broadcast?
        val lossAggregator = sampledTreePoints.treeAggregate(
          new LossAggregator(featureIndicesBundle, workingModel, currentRoot, metaData, loss))(
          seqOp = (agg, treePoint) => agg.add(treePoint),
          combOp = (agg1, agg2) => agg1.merge(agg2))

        val nodesToEnqueue = nodeBatch.indices.flatMap { nodeIdx =>
          val node = nodeBatch(nodeIdx)
          node.idxInBatch = None

          val bestSplitInfo = findBestSplit(lossAggregator.stats(nodeIdx),
            lossAggregator.featureIndicesBundle(nodeIdx), lossAggregator.offsets(nodeIdx), $(lambda), $(alpha))

          node.split = Some(bestSplitInfo.split)
          node.gain = Some(bestSplitInfo.gain)

          val leftChild = new WorkingNode(node.depth + 1)
          leftChild.prediction = Some(bestSplitInfo.leftPrediction * $(eta))
          leftChild.weight = Some(bestSplitInfo.leftWeight)
          node.leftChild = Some(leftChild)

          val rightChild = new WorkingNode(node.depth + 1)
          rightChild.prediction = Some(bestSplitInfo.rightPrediction * $(eta))
          rightChild.weight = Some(bestSplitInfo.rightWeight)
          node.rightChild = Some(rightChild)

          Iterator(leftChild, rightChild).filter {workingNode =>
              workingNode.depth < $(maxDepth) && workingNode.weight.get >= $(minInstanceWeight)
            }
        }
        nodeQueue ++= nodesToEnqueue
      }

      prune(currentRoot, $(gamma))

      if (!currentRoot.isLeaf) {
        workingModel.trees = workingModel.trees :+ currentRoot
        treeIdx += 1
      } else {
        /*
          If there is no sampling in records or features,
          then the next iteration will still be an empty tree.
          So skip to the end of the while loop.
         */
        if ($(sampleRatio) == 1.0 && $(featureSampleRatio) == 1.0) {
          treeIdx = $(numTrees)
        }
      }
    }

    workingModel.toTrainedModel(splits)
  }

  private def dequeueWithinMemLimit(queue: mutable.Queue[WorkingNode]): Array[WorkingNode] = {
    val arrayBuilder = mutable.ArrayBuilder.make[WorkingNode]()

    var idx: Int = 0
    while (queue.nonEmpty && idx < $(maxConcurrentNodes)){
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
      alpha: Double): SplitInfo = {
    val statsViews = createStatsViews(stats, featureIndices, offsets)

    val candidateSplit = (statsViews zip featureIndices) map { case (statsView, featureIdx)=>
      findBestSplitForSingleFeature(statsView, featureIdx, lambda, alpha)
    }

    candidateSplit.maxBy(_.gain)
  }

  private[sxgboost] def getCuSumAndTotal(ds: Seq[Double]) = {
    val cusum: Seq[Double] = ds.scan(0.0)(_+_)
    (cusum.drop(1).dropRight(1), cusum.last)
  }


  private[sxgboost] def sampleFeatureIndices(numFeatures: Int, featureSampleRatio: Double, numSamples: Int, rng: Random): Array[Array[Int]] = {
    val numSampledFeatures = Math.ceil(numFeatures * featureSampleRatio).toInt
    val indices = Range(0, numFeatures).toBuffer
    val arrayBuilder = mutable.ArrayBuilder.make[Array[Int]]()
    Range(0, numSamples).foreach { i =>
      arrayBuilder += rng.shuffle(indices).take(numSampledFeatures).toArray
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

  private[sxgboost] def prune(workingNode: WorkingNode, gamma: Double): Unit = {
    if (! workingNode.isLeaf) {
      prune(workingNode.leftChild.get, gamma)
      prune(workingNode.rightChild.get, gamma)

      if (workingNode.leftChild.get.isLeaf &&
        workingNode.rightChild.get.isLeaf &&
        workingNode.gain.get <= gamma) {
        workingNode.collapse()
      }
    }
  }
}


case class TrainedModel(bias: Double, trees: List[Node])

trait SparkXGBoostModelPredictor {
  def getModelPrediction(features: Vector, bias: Double, trees: List[Node], loss: Loss): Double = {
    val score = if (trees.nonEmpty){
      bias + trees.map{ node => node.predict(features) }.sum
    } else {
      bias
    }
    loss.toPrediction(score)
  }
}