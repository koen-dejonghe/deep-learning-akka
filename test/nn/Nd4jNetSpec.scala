package nn

import org.nd4j.linalg.factory.Nd4j._
import org.scalatestplus.play.PlaySpec

import scala.language.postfixOps
import scala.util.Random
class Nd4jNetSpec extends PlaySpec {

  "A NeuralNet" must {
    "initialize weights and biases" in {
      val topology = List(784, 15, 24, 10)
      val nn = new Nd4jNet(topology)

      nn.biases.size mustBe topology.size - 1
      nn.weights.size mustBe topology.size - 1

      nn.biases.zipWithIndex.foreach { case (v, i) =>
        v.length mustBe topology(i+1)
      }

      topology.sliding(2).zipWithIndex.foreach { case (t, i) =>
        val x = t.head
        val y = t(1)

        nn.weights(i).shape()(0) mustBe y
        nn.weights(i).shape()(1) mustBe x

      }

    }

    "feed forward" in {
      val topology = List(2, 3, 10)
      val nn = new Nd4jNet(topology)
      val x = ones(2, 1)

      val (zs, as) = nn.feedForward(x)

      println(zs)
      println(as)
    }

    "back prop" in {
      val topology = List(3, 4, 10)
      val nn = new Nd4jNet(topology)
      val x = ones(3).transpose()

      val (nb, nw) = nn.backProp(x, Nd4jNet.oneHotEncoded(3))

      println(nb)
      println(nw)

      nb.size mustBe topology.size - 1
      nw.size mustBe topology.size - 1

      nb.zipWithIndex.foreach { case (v, i) =>
        v.length mustBe topology(i+1)
      }

      topology.sliding(2).zipWithIndex.foreach { case (t, i) =>
        val x = t.head
        val y = t(1)
        nw(i).shape()(0) mustBe y
        nw(i).shape()(1) mustBe x
      }
    }

    "mini batch" in {
      val topology = List(3, 4, 10)
      val nn = new Nd4jNet(topology)

      println("========> before")
      println(nn.biases)
      println(nn.weights)

      val miniBatch = (1 to 5).map { _ =>
        (rand(3, 1), Nd4jNet.oneHotEncoded(Random.nextInt(10)))
      } toList

      nn.updateMiniBatch(miniBatch, 3.0)

      println("========> after")
      println(nn.biases)
      println(nn.weights)
    }

    "sgd" in {
      val topology = List(3, 4, 10)
      val nn = new Nd4jNet(topology)

      println("========> before")
      println(nn.biases)
      println(nn.weights)

      val trainingData = (1 to 10).map { _ =>
        (rand(3, 1), Nd4jNet.oneHotEncoded(Random.nextInt(10)))
      } toList

      val epochs = 3
      val batchSize = 2
      val learningRate = 3.0

      nn.sgd(trainingData, epochs, batchSize, learningRate)

      println("========> after")
      println(nn.biases)
      println(nn.weights)
    }

    "run" in {

      val topology = List(784, 15, 10)
      val nn = new Nd4jNet(topology)
      val epochs = 3
      val batchSize = 100
      val learningRate = 3.0
      val trainingData = Nd4jNet.loadData("data/mnist_train.csv")
      nn.sgd(trainingData, epochs, batchSize, learningRate)

    }

  }

}
