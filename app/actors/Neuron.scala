package actors

import math.Vector
import akka.actor.{Actor, ActorRef, Props}

class Neuron(targetLayer: List[ActorRef],
             nodeId: Int,
             weights: Vector,
             bias: Double,
             activation: (Double) => Double)
    extends Actor {

  override def receive: Receive = collect(weights, bias, Map.empty[Int, Double])

  def collect(weights: Vector, bias: Double, xs: Map[Int, Double]): Receive = {

    case Weights(ws) =>
      context.become(collect(ws, bias, xs))

    case Bias(bs) =>
      context.become(collect(weights, bs, xs))

    case Output(n, x) =>
      val m = xs.updated(n, x)
      require(weights.size >= m.size)
      if (m.size == weights.size) {
        fire(Vector(m), weights, bias)
        context.become(collect(weights, bias, Map.empty[Int, Double]))
      } else {
        context.become(collect(weights, bias, m))
      }
  }

  def fire(xs: Vector, weights: Vector, bias: Double): Unit = {
    val output = activation(xs * weights + bias)
    targetLayer.foreach(n => n ! Output(nodeId, output))
  }

}

object Neuron {
  def props(targetLayer: List[ActorRef],
            nodeId: Int,
            weights: Vector,
            bias: Double,
            activation: (Double) => Double): Props =
    Props(new Neuron(targetLayer, nodeId, weights, bias, activation))
}

case class Weights(weights: Vector)
case class Bias(bias: Double)
case class Output(node: Int, value: Double)
