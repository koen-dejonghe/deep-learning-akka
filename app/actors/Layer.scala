package actors

import akka.actor.{Actor, ActorRef}
import org.nd4j.linalg.api.ndarray.{INDArray => Matrix}
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

case class Shape(n: Int, m: Int)

class Layer(shape: Shape, fwd: ActorRef, bwd: ActorRef) extends Actor {

  val weights: Matrix = randn(shape.m, shape.n) / Math.sqrt(shape.n)
  val biases: Matrix = randn(shape.m, 1)
  val activations: Matrix = zeros(shape.m, 1)
  val nablaWeights: Matrix = zeros(shape.m, shape.n)
  val nablaBiases: Matrix = zeros(shape.m, 1)

  override def receive: Receive = {

    case FeedForward(x) =>
      val z = (weights dot x) + biases
      val a = sigmoid(z)
      activations.assign(a)
      fwd ! FeedForward(a)

    case DeltaBackward(a, d) =>
      val sp = derivative(a)
      val delta = (weights.transpose() dot d) * sp

      val nb = delta
      val nw = delta dot activations.transpose()

      nablaBiases += nb
      nablaWeights += nw

      bwd ! DeltaBackward(activations, delta)

    case UpdateWeightsAndBiases(miniBatchSize, learningRate, lambda, trainingDataSize) =>
      val m = miniBatchSize
      val lm = learningRate / m
      val lln = 1.0 - learningRate * (lambda / trainingDataSize)

      biases -= (nablaBiases * lm)
      weights *= lln
      weights -= nablaWeights * lm

  }

  def derivative(z: Matrix): Matrix = z * (-z + 1.0)
}

case class FeedForward(x: Matrix)
case class DeltaBackward(a: Matrix, delta: Matrix)
case class UpdateWeightsAndBiases(miniBatchSize: Int, learningRate: Double, lambda: Double, trainingDataSize: Int)
