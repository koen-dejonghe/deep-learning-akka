package nn

import breeze.linalg.DenseVector
import org.scalatestplus.play.PlaySpec

import org.nd4j.linalg.factory.Nd4j._
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

    "one hot encode" in {
      val ohe = NeuralNet.oneHotEncoded(3)
      ohe mustBe DenseVector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
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
  }

}
