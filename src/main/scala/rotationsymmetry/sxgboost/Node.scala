package rotationsymmetry.sxgboost

import org.apache.spark.mllib.linalg.Vector

private[sxgboost] abstract class Node extends Serializable {

  def predict(features: Vector): Double
}

private[sxgboost] class InnerNode(val split: Split, val leftChild: Node, val rightChild: Node) extends Node {

  override def predict(features: Vector): Double = {
    if (split.shouldGoLeft(features)) {
      leftChild.predict(features)
    } else {
      rightChild.predict(features)
    }
  }
}

private[sxgboost] class LeafNode(val prediction: Double) extends Node {

  override def predict(features: Vector): Double = prediction
}
