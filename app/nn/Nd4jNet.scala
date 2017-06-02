package nn

import java.io.{BufferedInputStream, FileInputStream}
import java.util.zip.GZIPInputStream

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._

import scala.io.Source
import scala.language.postfixOps
import scala.util.Random

class Nd4jNet(topology: List[Int]) {

//  DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)

  /*
  CudaEnvironment
    .getInstance()
    .getConfiguration
    .setPoolSize(24)
    .setVerbose(true)
   */

  val biases: List[INDArray] =
    topology.tail.map(size => randn(size, 1))

  val weights: List[INDArray] =
    topology.sliding(2).map(t => randn(t(1), t.head)).toList

  /**
    * Train the neural network using mini-batch stochastic
    * gradient descent.  The ``training_data`` is a list of tuples
    * ``(x, y)`` representing the training inputs and the desired
    * outputs.
    */
  def sgd(trainingData: List[(INDArray, INDArray)],
          epochs: Int,
          miniBatchSize: Int,
          learningRate: Double,
          testData: List[(INDArray, Int)] = List.empty): Unit = {

    (1 to epochs).foreach { j =>
      println(s"Epoch $j starting")
      val shuffled = Random.shuffle(trainingData)
      shuffled.sliding(miniBatchSize, miniBatchSize).foreach { miniBatch =>
        updateMiniBatch(miniBatch, learningRate)
      }

      if (testData.nonEmpty) {
        val eval = evaluate(testData)
        println(s"Epoch $j ==> $eval")
      } else {
        println(s"Epoch $j complete")
      }
    }
  }

  def feedForward(x: INDArray): List[INDArray] = {
    biases.zip(weights).foldLeft(List(x)) {
      case (as, (b, w)) =>
        // z = np.dot(w, activation)+b
        val z = w.mmul(as.last).add(b)
        val a = sigmoid(z)
        as :+ a
    }
  }

  /**
    * Update the network's weights and biases by applying
    * gradient descent using backpropagation to a single mini batch.
    * The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
    * is the learning rate
    */
  def updateMiniBatch(miniBatch: List[(INDArray, INDArray)],
                      learningRate: Double): Unit = {

    val nablaBiases = biases.map(b => zeros(b.shape(): _*))
    val nablaWeights = weights.map(w => zeros(w.shape(): _*))

    miniBatch.foreach {
      case (x, y) =>
        val (deltaNablaB, deltaNablaW) = backProp(x, y)

        nablaBiases.zip(deltaNablaB).foreach {
          case (nb, dnb) =>
            nb.addi(dnb)
        }

        nablaWeights.zip(deltaNablaW).foreach {
          case (nw, dnw) =>
            nw.addi(dnw)
        }
    }

    biases.zip(nablaBiases).foreach {
      case (b, nb) =>
        b.subi(nb.mul(learningRate / miniBatch.size))
    }

    weights.zip(nablaWeights).foreach {
      case (w, nw) =>
        w.subi(nw.mul(learningRate / miniBatch.size))
    }

  }

  /**
    * Return a tuple ``(nabla_b, nabla_w)`` representing the
    * gradient for the cost function C_x.  ``nabla_b`` and
    * ``nabla_w`` are layer-by-layer lists of arrays, similar
    * to ``biases`` and ``weights``.
    */
  def backProp(x: INDArray, y: INDArray): (List[INDArray], List[INDArray]) = {

    val activations = feedForward(x)

    // delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
    val a = activations.last.sub(y)
    val sp = derivative(activations.last)
    val delta = a.mul(sp)

    val inb = delta
    // np.dot(delta, activations[-2].transpose())
    val inw = delta.mmul(activations(activations.size - 2).transpose())

    val (nablaBiases, nablaWeights) = (2 until topology.size)
      .foldLeft((List(inb), List(inw))) {
        case ((nbl, nwl), l) =>
          // z = zs[-l]
          val sp = derivative(activations(activations.size - l))

          // delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
          val delta = weights(weights.size - l + 1)
            .transpose()
            .mmul(nbl.head) // last added nb to nbl is the previous delta
            .mul(sp)

          val nb = delta
          // nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
          val nw =
            delta.mmul(activations(activations.size - l - 1).transpose())

          (nb :: nbl, nw :: nwl)
      }

    (nablaBiases, nablaWeights)
  }

  /**
    * Derivative of the sigmoid function.
    */
  def derivative(z: INDArray): INDArray = {
    // sigmoid(z)*(1-sigmoid(z))
    z.mul(z.neg().add(1.0))
  }

  def evaluate(testData: List[(INDArray, Int)]): Double = {
    val correct = testData.foldLeft(0.0) {
      case (t, (x, y)) =>
        val activation = biases.zip(weights).foldLeft(x) {
          case (a, (b, w)) =>
            sigmoid(w.mmul(a).add(b))
        }

        val guess = argMax(activation).getInt(0)

        if (guess == y) t + 1 else t
    }
    correct / testData.size.toDouble
  }

}

object Nd4jNet {

  def oneHotEncoded(x: Int, base: Int = 10): INDArray = {
    val v = zeros(1, base)
    v.putScalar(x, 1.0)
  }

  def gzis(fname: String): GZIPInputStream =
    new GZIPInputStream(new BufferedInputStream(new FileInputStream(fname)))

  def loadData(fname: String): List[(INDArray, Int)] = {
    Source.fromInputStream(gzis(fname)).getLines() map { line =>
      val tokens = line.split(",")
      val (y, x) = (tokens.head.toInt, tokens.tail.map(_.toDouble / 255.0))
      (create(x).transpose(), y)
    } toList
  }

  def main(args: Array[String]) {
    println("Hello, world")

    val topology = args.map(_.toInt).toList
    val nn = new Nd4jNet(topology)
    val epochs = 30
    val batchSize = 10
    val learningRate = 3.0
    val trainingData = Nd4jNet.loadData("data/mnist_train.csv.gz").map {
      case (x, y) => (x, oneHotEncoded(y))
    }
    val testData = Nd4jNet.loadData("data/mnist_test.csv.gz")
    nn.sgd(trainingData, epochs, batchSize, learningRate, testData)
  }

}
