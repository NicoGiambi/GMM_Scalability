name := "GMM_Scalability"

version := "0.2"

scalaVersion := "2.12.10"

libraryDependencies += "org.apache.spark" %% "spark-core" % "3.1.3"

libraryDependencies += "org.apache.spark" %% "spark-mllib-local" % "3.1.3"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.1.3"

libraryDependencies += "com.lihaoyi" %% "upickle" % "1.5.0"

libraryDependencies += "com.lihaoyi" %% "os-lib" % "0.8.1"

unmanagedJars in Compile += file("lib/gradientgmm_2.12-1.0.jar")