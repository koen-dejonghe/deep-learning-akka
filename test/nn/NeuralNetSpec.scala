package nn

import breeze.linalg.DenseVector
import org.scalatestplus.play.PlaySpec

class NeuralNetSpec extends PlaySpec {

  "A NeuralNet" must {
    "initialize weights and biases" in {
      val topology = List(2, 3, 10)
      val nn = new NeuralNet(topology)

      nn.biases.size mustBe topology.size - 1
      nn.weights.size mustBe topology.size - 1

      nn.biases.zipWithIndex.foreach { case (v, i) =>
        v.length mustBe topology(i+1)
      }

      topology.sliding(2).zipWithIndex.foreach { case (t, i) =>
        val x = t.head
        val y = t(1)

        nn.weights(i).rows mustBe y
        nn.weights(i).cols mustBe x

      }

    }

    "feed forward" in {
      val topology = List(2, 3, 10)
      val nn = new NeuralNet(topology)
      val x = DenseVector.ones[Double](2)

      val (zs, as) = nn.feedForward(x)

      println(zs.size)
      println(as.size)
    }

    "one hot encode" in {
      val ohe = NeuralNet.oneHotEncoded(3)
      ohe mustBe DenseVector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }

    "back prop" in {
      val topology = List(2, 3, 10)
      val nn = new NeuralNet(topology)
      val x = DenseVector.ones[Double](2)

      val (nb, nw) = nn.backProp(x, NeuralNet.oneHotEncoded(3))

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
        nw(i).rows mustBe y
        nw(i).cols mustBe x
      }

    }
  }

}
