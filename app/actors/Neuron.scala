package actors

import math._
import akka.actor.{Actor, ActorLogging, ActorRef, Props}

import scala.language.postfixOps

class Neuron(targetLayer: List[ActorRef],
             nodeId: Int,
             learningRate: Double,
             weights: Vector,
             bias: Double,
             activation: (Double) => Double)
    extends Actor
    with ActorLogging {

  override def receive: Receive =
    feedForward(weights, bias, Map.empty, Map.empty)

  def feedForward(weights: Vector,
                  bias: Double,
                  sourceLayer: Map[Int, ActorRef],
                  activationMap: Map[Int, Double] = Map.empty): Receive = {

    case Activation(n, x) =>
//      log.debug(s"received activation: $n, $x")

      val sl: Map[Int, ActorRef] = sourceLayer.updated(n, sender()) // dynamically collect source layer
      val m = activationMap.updated(n, x)
      require(weights.size >= m.size)
      if (m.size == weights.size) {
        val activations = Vector(m)
        val output = fire(activations, weights, bias)
        context.become(backProp(weights, bias, sl, output))
      } else {
        context.become(feedForward(weights, bias, sl, m))
      }

    case unknown =>
      log.debug(s"unknown in ff: $unknown")

  }

  def backProp(weights: Vector,
               bias: Double,
               sourceLayer: Map[Int, ActorRef],
               output: Double,
               deltaSet: List[Double] = List.empty): Receive = {
    case Delta(n, d) =>
      val m = d :: deltaSet
      if (m.size == targetLayer.size) {
        val deltas = Vector(m)

        val sp = derivative(output)
        val delta = weights.sum * deltas.sum * sp

        val newBias = bias - delta
        val newWeights = weights - delta

        log.debug(s"senders: ${sourceLayer.size}")

        sourceLayer.foreach { case (n, ref) =>
          ref ! Delta(n, delta)
        }
        context.become(feedForward(newWeights, newBias, sourceLayer))
      } else {
        context.become(backProp(weights, bias, sourceLayer, output, m))
      }

    case unknown =>
      log.debug(s"unknown in bp: $unknown")
  }

  def fire(xs: Vector, weights: Vector, bias: Double): Double = {
    val output = activation(xs.dot(weights) + bias)
    targetLayer.foreach(n => n ! Activation(nodeId, output))
    output
  }

}

object Neuron {
  def props(targetLayer: List[ActorRef],
            nodeId: Int,
            learningRate: Double,
            weights: Vector,
            bias: Double,
            activation: (Double) => Double): Props =
    Props(
      new Neuron(targetLayer, nodeId, learningRate, weights, bias, activation))
}

case class Activation(node: Int, value: Double)
case class Delta(node: Int, delta: Double)
