name := "SparkXGBoost"

version := "0.1"

scalaVersion := "2.10.6"

val testSparkVersion = settingKey[String]("The version of Spark to test against.")

testSparkVersion := sys.props.getOrElse("spark.testVersion", "1.5.1")

parallelExecution := false

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.10" % testSparkVersion.value,
  "org.apache.spark" % "spark-mllib_2.10" % testSparkVersion.value,
  "org.scalatest" % "scalatest_2.10" % "2.2.4" % "test",
  "com.databricks" % "spark-csv_2.10" % "1.2.0" % "test"
)
