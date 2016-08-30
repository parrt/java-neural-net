package nn;

public class Particle {
	public Network best;
	public double bestScore;
	public Particle(Network net) { this.best = net; }
}
