name := "SparkXGBoost"

version := "0.1"

scalaVersion := "2.10.6"

parallelExecution := false

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.10" % "1.5.1",
  "org.apache.spark" % "spark-mllib_2.10" % "1.5.1",
  "org.scalatest" % "scalatest_2.10" % "2.2.4" % "test"
)
