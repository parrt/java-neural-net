package nn;

import lib.StdRandom;

public class Particle {
	public Network position;
	public double velocity;
	public Network best;
	public double bestScore;

	public Particle(Network position, double bestScore) {
		this.position = position;
		this.best = position;
		this.bestScore = bestScore;
		this.velocity = StdRandom.uniform();
	}
}
