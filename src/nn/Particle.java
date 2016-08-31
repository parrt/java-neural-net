package nn;

import lib.StdRandom;

public class Particle {
	public double velocity;
	public Network best;
	public double bestScore;

	public Particle(Network best, double bestScore) {
		this.best = best;
		this.bestScore = bestScore;
		this.velocity = StdRandom.uniform();
	}
}
