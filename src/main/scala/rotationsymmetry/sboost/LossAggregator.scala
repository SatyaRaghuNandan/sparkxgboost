package rotationsymmetry.sboost

class LossAggregator(
      val featureIndicesBundle: Array[Array[Int]],
      val workingModel: WorkingModel,
      val currentRoot: WorkingNode,
      val metaData: MetaData,
      val loss: Loss) {

  val (offsets: Array[Array[Int]], stats: Array[Array[Double]]) = {
    val offsetsAndStats = featureIndicesBundle map {indices: Array[Int]=>
      val sizes: Array[Int] = indices.map(i => metaData.numBins(i)*2)
      val offsetsAndTotalSize = sizes.scan(0)(_+_)
      (offsetsAndTotalSize.dropRight(1), Array.fill[Double](offsetsAndTotalSize.last)(0))
    }

    (offsetsAndStats.map(_._1), offsetsAndStats.map(_._2))
  }

  def add(treePoint: TreePoint): LossAggregator = {
    val yhat0 = workingModel.predict(treePoint)
    val diff1 = loss.diff1(treePoint.label, yhat0)
    val diff2 = loss.diff2(treePoint.label, yhat0)

    if (currentRoot.locateNode(treePoint).idxInBatch.isDefined) {
      val nodeIdx = currentRoot.locateNode(treePoint).idxInBatch.get
      var i: Int = 0
      while (i < featureIndicesBundle.length){
        val idx = featureIndicesBundle(nodeIdx)(i)
        val bin = treePoint.binnedFeature(idx)
        // offset for the bin in the feature
        val statsOffset = offsets(nodeIdx)(i) + bin * 2
        stats(nodeIdx)(statsOffset) += diff1
        stats(nodeIdx)(statsOffset + 1) += diff2
        i +=1
      }
    }

    this
  }

  def merge(that: LossAggregator): LossAggregator = {
    var i: Int = 0
    while (i < stats.length){
      var j: Int = 0
      while (j < stats(i).length) {
        stats(i)(j) += that.stats(i)(j)
        j += 1
      }
      i += 1
    }
    this
  }
}
