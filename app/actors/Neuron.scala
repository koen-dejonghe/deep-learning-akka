package actors

import math.Vector
import akka.actor.{Actor, ActorRef}

class Neuron(targets: List[ActorRef], activation: (Double) => Double) extends Actor {

  override def receive: Receive = collect(Vector(), 0.0, Vector())

  def collect(weights: Vector, bias: Double, xs: Vector): Receive = {
    case Weights(ws) => context.become(collect(ws, bias, xs))
    case Bias(bs) => context.become(collect(weights, bs, xs))
    case x: Double =>
      val v = xs :+ x
      require(weights.size >= v.size)
      if (v.size == weights.size) {
        fire(v, weights, bias)
        context.become(collect(weights, bias, Vector()))
      }
      else {
        context.become(collect(weights, bias, v))
      }
  }

  def fire(xs: Vector, weights: Vector, bias: Double): Unit = {
    val output = activation(xs * weights + bias)
    targets.foreach(t => t ! output)
  }

}

case class Weights(weights: Vector)
case class Bias(bias: Double)
