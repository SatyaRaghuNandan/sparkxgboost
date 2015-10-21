package rotationsymmetry.sxgboost

import org.apache.spark.mllib.linalg.Vector

abstract class Node {

  def predict(features: Vector): Double
}

class InnerNode(val split: Split, val leftChild: Node, val rightChild: Node) extends Node {

  override def predict(features: Vector): Double = {
    if (split.shouldGoLeft(features)) {
      leftChild.predict(features)
    } else {
      rightChild.predict(features)
    }
  }
}

class LeafNode(val prediction: Double) extends Node {

  override def predict(features: Vector): Double = prediction
}
