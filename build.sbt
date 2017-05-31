name := """deep-learning-akka"""

version := "1.0-SNAPSHOT"

lazy val root = (project in file(".")).enablePlugins(PlayScala)

scalaVersion := "2.11.11"

libraryDependencies ++= Seq(
  jdbc,
  cache,
  ws,
  "org.scalatestplus.play" %% "scalatestplus-play" % "1.5.1" % Test
)

libraryDependencies += "com.typesafe.akka" %% "akka-testkit" % "2.5.2"

libraryDependencies += "org.scalanlp" %% "breeze" % "0.13.1"
libraryDependencies += "org.scalanlp" %% "breeze-natives" % "0.13.1"


classpathTypes += "maven-plugin"


//libraryDependencies += "org.nd4j" % "nd4j-native" % "0.4-rc3.10" classifier "" classifier "linux-x86_64"

libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "0.8.0"
//libraryDependencies += "org.nd4j" % "nd4j-cuda-8.0" % "0.8.0"
//libraryDependencies += "org.nd4j" % "nd4j-cuda-8.0-platform" % "0.8.0"
libraryDependencies += "org.nd4j" %% "nd4s" % "0.8.0"




