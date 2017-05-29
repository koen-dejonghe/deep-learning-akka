package actors

import akka.actor.ActorSystem
import akka.testkit.{ImplicitSender, TestKit, TestProbe}
import math.Vector
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpecLike}

class NetworkSpec extends TestKit(ActorSystem("NetworkSpec")) with ImplicitSender
  with WordSpecLike with Matchers with BeforeAndAfterAll {

  override def afterAll {
    TestKit.shutdownActorSystem(system)
  }

  "A Network Actor" must {
    "forward propagate" in {

      val probe = TestProbe()

      val topology = List(784, 15, 10)
      val network = system.actorOf(Network.props(topology, probe.ref))
      val v = Vector.randn(784)
      network ! v
      val output = probe.receiveN(10)
      probe.expectNoMsg()
      println(output)
    }

    "forward propagate again" in {
      val probe = TestProbe()
      val topology = List(1, 1)
      val network = system.actorOf(Network.props(topology, probe.ref))
      network ! Vector(0.1)
      val output = probe.receiveN(1).head.asInstanceOf[Output]
      println(output)
      network ! Vector(output.value)
      val output2 = probe.receiveN(1).head.asInstanceOf[Output]
      println(output2)
    }
  }

}
