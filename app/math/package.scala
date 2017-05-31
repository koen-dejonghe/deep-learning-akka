/**
  * Created by koen on 2017/05/30.
  */
package object math {

  def sigmoid(z: Double): Double = 1.0 / (1.0 + Math.exp(-z))

  def derivative(z: Double): Double = z * (1.0 - z)

  def costDerivative(outputActivations: Vector, y: Vector): Vector = outputActivations - y

}
