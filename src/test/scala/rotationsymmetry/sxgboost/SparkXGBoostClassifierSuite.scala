package rotationsymmetry.sxgboost

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.functions.udf
import org.scalatest.FunSuite
import rotationsymmetry.sxgboost.loss.LogisticLoss
import rotationsymmetry.sxgboost.utils.MLlibTestSparkContext

class SparkXGBoostClassifierSuite extends FunSuite with TestData with MLlibTestSparkContext {

  test("test with simple data") {
    val rawdata = Seq(
      LabeledPoint(0, Vectors.dense(0.0, 0.0)),
      LabeledPoint(0, Vectors.dense(0.0, 0.0)),
      LabeledPoint(1, Vectors.dense(0.0, 0.0)),

      LabeledPoint(1, Vectors.dense(1.0, 0.0)),
      LabeledPoint(1, Vectors.dense(1.0, 0.0)),
      LabeledPoint(0, Vectors.dense(1.0, 0.0)),

      LabeledPoint(1, Vectors.dense(0.0, 1.0)),
      LabeledPoint(1, Vectors.dense(0.0, 1.0)),
      LabeledPoint(0, Vectors.dense(0.0, 1.0)),

      LabeledPoint(0, Vectors.dense(1.0, 1.0)),
      LabeledPoint(0, Vectors.dense(1.0, 1.0)),
      LabeledPoint(1, Vectors.dense(1.0, 1.0))
    )

    val data = sqlContext.createDataFrame(sc.parallelize(rawdata, 2))

    val truthUDF = udf { feature: Vector =>
      if (feature(0) == feature(1))
        0.0
      else
        1.0
    }

    val dataWithTruth = data.withColumn("truth", truthUDF(data("features")))

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(2)
      .fit(data)

    val sparkXGBoostClassifier = new SparkXGBoostClassifier(new LogisticLoss)
      .setFeaturesCol("indexedFeatures")
      .setMaxDepth(2)
      .setNumTrees(1)
    val sparkXGBoostPipeline = new Pipeline()
      .setStages(Array(featureIndexer, sparkXGBoostClassifier))
    val sXGBoostModel = sparkXGBoostPipeline.fit(data)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("truth")
      .setPredictionCol("prediction")
      .setMetricName("precision")

    val precision = evaluator.evaluate(sXGBoostModel.transform(dataWithTruth))

    assert(precision === 1.0)
  }
}
