package nn;

import lib.StdRandom;

public class Particle {
	public Network position;
	public double velocity;
	public Network best;
	public double lowestCost;

	public Particle(Network position, double lowestCost) {
		this.position = position;
		this.best = position;
		this.lowestCost = lowestCost;
		this.velocity = StdRandom.uniform();
	}
}
