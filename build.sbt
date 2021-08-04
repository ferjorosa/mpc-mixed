name := "multi-partition-mixed"

version := "0.1"

scalaVersion := "2.13.6"

libraryDependencies ++= Seq (
  "colt" % "colt" % "1.2.0",
  "commons-cli" % "commons-cli" % "1.2",
  "org.apache.commons" % "commons-lang3" % "3.6",
  "org.apache.commons" % "commons-math3" % "3.6.1",
  "org.slf4j" % "slf4j-simple" % "1.7.26",
  "com.google.guava" % "guava" % "27.1-jre",
  "nz.ac.waikato.cms.weka" % "weka-stable" % "3.8.1",
  "com.google.code.gson" % "gson" % "2.8.6"
)
