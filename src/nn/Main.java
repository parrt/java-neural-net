package nn;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {
	public static final int NUM_PARTICLES = 10;
	public static final int NUM_ITERATIONS = 100;

	public static final int IMAGE_LEN = 784;
	public static final int HIDDEN_LAYER_LEN = 15;
	public static final int OUTPUT_LAYER_LEN = 10;

	static double[][] X;
	static int[] labels;
	static double[][] onehots;

	public static void main(String[] args) throws IOException {
		List<Image> training = MNISTLoader.loadTrainingImages("/Users/parrt/data/mnist");
		System.out.printf("Loaded %d training images\n", training.size());
//		List<Image> testing = MNISTLoader.loadTestingImages("/Users/parrt/data/mnist");
//		System.out.printf("Loaded %d testing images\n", testing.size());

		training = training.subList(0, 20);
		System.out.printf("Using %d images\n", training.size());

		X = new double[training.size()][];
		labels = new int[training.size()];
		onehots = new double[training.size()][];
		for (int i = 0; i<X.length; i++) {
			X[i] = training.get(i).data;
			labels[i] = training.get(i).label;
			onehots[i] = new double[OUTPUT_LAYER_LEN];
			onehots[i][labels[i]] = 1.0;
		}
//		bareBonesParticleSwarmOptimizer(training);
//		slightlyGuidedRandomSearchOptimizer(training);
//		tinySlightlyGuidedRandomSearchOptimizer();
		gradientDescentFiniteDifference(training);
	}

	/**
	 * Use a simple gradient descent approach to find weights
	 */
	public static Network gradientDescentFiniteDifference(List<Image> training) {
	/*
	LEARNING_RATE = 10
	h = 0.0001
	PRECISION = 0.0000001 # can't be too small as f(x)-f(xprev) prec is low
	def f(x): return np.cos(3*np.pi*x) / x
	x0s = [runif_(.1,1.2), runif_(.1,1.2)] # random starting positions
	tracex = minimize(f, x0s[0], LEARNING_RATE, h, PRECISION)
	 */
		double EPSILON = 2.22e-16;
		double h = 0.001;
		double eta = 1.0;
		Network prev_pos = null;
		Network p = new Network(0.0, 1.0, IMAGE_LEN, HIDDEN_LAYER_LEN, OUTPUT_LAYER_LEN);
		while ( true ) {
			prev_pos = p;
			Network finite_diff = p.finiteDifference(h, X, onehots);
			p = p.subtract(finite_diff.scale(eta));
			// print "f(%1.12f) = %1.12f" % (x, f(x)),
			double delta = p.cost(X, onehots) - prev_pos.cost(X, onehots);
			int correct = p.fitness(X, labels);
			System.out.printf("cost %3.5f, correct %d\n", p.cost(X, onehots), correct);
			// print ", delta = %1.20f" % delta
			// stop when small change in vertical but not heading down
			if ( delta>=0 && Math.abs(delta)<0.000001 ) {
				break;
			}
		}
		return p;
	}

	public static void bareBonesParticleSwarmOptimizer(List<Image> training) {
		double w = 0.729844;
		double c1 = 1.49618, c2 = 1.49618;

		// init all particles
		Particle[] particles = new Particle[NUM_PARTICLES];
		double min = Double.MAX_VALUE;
		int mini = -0;
		for (int j = 0; j<NUM_PARTICLES; j++) {
			Network net = new Network(0.0, 1.0, IMAGE_LEN, HIDDEN_LAYER_LEN, OUTPUT_LAYER_LEN);
			double cost = net.cost(X, onehots);
			particles[j] = new Particle(net, cost);
			if ( cost<min ) {
				min = cost;
				mini = j;
			}
			int correct = net.fitness(X, labels);
			System.out.printf("%d: cost %3.3f, num correct %d %2.2f%%\n",
			                  j, cost, correct, 100*correct/(float) training.size());
		}

		Particle global = particles[mini];
		System.out.println("gbest index is "+mini+": "+global.lowestCost);

		for (int i = 0; i<NUM_ITERATIONS; i++) {
			System.out.println("ITERATION "+i);
			Network genBest = null;
			double genLowestCost = Double.MAX_VALUE;
			for (int j = 0; j<NUM_PARTICLES; j++) {
				Particle p = particles[j];
				Network mu = (p.best.add(global.best)).scale(0.5);
//				Network sigma = (p.best.subtract(global.best)).abs();
				Network sigma = Network.ones(784, 15, 10).scale(1.0);
				p.position = new Network(mu, sigma, 784, 15, 10);
				double cost = p.position.cost(X, onehots);
				int correct = p.position.fitness(X, labels);
				System.out.printf("%d: cost=%3.2f, num correct %d %2.2f%%\n",
				                  j, cost, correct, 100*correct/(float) training.size());
				// Update this particle's best
				if ( cost<p.lowestCost ) {
					p.best = p.position;
					p.lowestCost = cost;
				}
				if ( cost<genLowestCost ) {
					genLowestCost = cost;
					genBest = p.position;
				}
//				System.out.println(p.position);
			}
			// Update best global particle
			if ( genLowestCost<global.lowestCost ) {
				global.best = genBest;
				global.lowestCost = genLowestCost;
			}
			int correct = global.best.fitness(X, labels);
			System.out.printf("global cost %3.2f, num correct %d %2.2f%%\n",
			                  global.lowestCost, correct, 100*correct/(float) training.size());
		}
	}

	public static void slightlyGuidedRandomSearchOptimizer(List<Image> training) {
		Network globalBest = new Network(0.0, 1.0, 784, 15, 10);
		double globalLowestCost = globalBest.cost(X, onehots);
		int correct = globalBest.fitness(X, labels);
		int correct2 = globalBest.fitness(X, labels);
		if ( correct!=correct2) {
			System.err.println("adsfkjabbcasdsfasdf");
		}
		System.out.printf("global lowest cost %3.2f, num correct %d %2.2f%%\n",
		                  globalLowestCost, correct, 100*correct/(float) training.size());
		Network mu = globalBest;
		Network sigma = Network.ones(784, 15, 10).scale(1.0);

		double learningRate = 1.5;

		for (int i = 0; i<NUM_ITERATIONS; i++) {
			System.out.println("ITERATION "+i);
			// find the best location in a generation
			Network genBest = null;
			double genLowestCost = Float.MAX_VALUE;
			for (int j = 0; j<NUM_PARTICLES; j++) {
				Network pos = new Network(mu, sigma, 784, 15, 10);
				double cost = pos.cost(X, onehots);
				correct = pos.fitness(X, labels);
				System.out.printf("%d: cost=%3.2f, num correct %d %2.2f%%\n",
				                  j, cost, correct, 100*correct/(float) training.size());
				if ( cost<genLowestCost ) {
					genBest = pos;
					genLowestCost = cost;
				}
			}
			System.out.printf("gen lowest cost %3.2f\n", genLowestCost);
			// Update best global particle
			if ( genLowestCost<globalLowestCost ) {
				// only move center if we have improved with this gen
				Network delta = genBest.subtract(globalBest);
				mu = globalBest.add(delta.scale(learningRate));
//				mu = genBest;
				sigma = delta.abs();
				globalBest = genBest;
				globalLowestCost = genLowestCost;
			}
			correct = globalBest.fitness(X, labels);
			System.out.printf("global lowest cost %3.2f, num correct %d %2.2f%%\n",
			                  globalLowestCost, correct, 100*correct/(float) training.size());
		}
	}

	public static void tinySlightlyGuidedRandomSearchOptimizer() {
		List<Image> training = new ArrayList<>();
		training.add(new Image(new double[]{0,0,0}, 0));
		training.add(new Image(new double[]{0,1,0}, 1));
		training.add(new Image(new double[]{1,0,0}, 1));
		training.add(new Image(new double[]{1,0,1}, 1));
		training.add(new Image(new double[]{1,1,1}, 0));
		double[][] X = new double[training.size()][];
		int[] labels = new int[training.size()];
		double[][] onehots = new double[training.size()][];
		for (int i = 0; i<X.length; i++) {
			X[i] = training.get(i).data;
			labels[i] = training.get(i).label;
			onehots[i] = new double[2];
			onehots[i][labels[i]] = 1.0;
		}

		Network globalBest = new Network(0.0, 1.0, 3, 2, 2);
		double globalLowestCost = globalBest.cost(X, onehots);
		int correct = globalBest.fitness(X, labels);
		int correct2 = globalBest.fitness(X, labels);
		if ( correct!=correct2) {
			System.err.println("adsfkjabbcasdsfasdf");
		}
		System.out.printf("global lowest cost %3.2f, num correct %d %2.2f%%\n",
		                  globalLowestCost, correct, 100*correct/(float) training.size());
		Network mu = globalBest;
		Network sigma = Network.ones(3, 2, 2).scale(1.0);

		double learningRate = 1.0;

		for (int i = 0; i<NUM_ITERATIONS; i++) {
			System.out.println("ITERATION "+i);
			// find the best location in a generation
			Network genBest = null;
			double genLowestCost = Float.MAX_VALUE;
			for (int j = 0; j<3; j++) {
				Network pos = new Network(mu, sigma, 3, 2, 2);
				double cost = pos.cost(X, onehots);
				correct = pos.fitness(X, labels);
				System.out.printf("%d: cost=%3.2f, num correct %d %2.2f%%\n",
				                  j, cost, correct, 100*correct/(float) training.size());
				if ( cost<genLowestCost ) {
					genBest = pos;
					genLowestCost = cost;
				}
			}
			System.out.printf("gen lowest cost %3.2f\n", genLowestCost);
			// Update best global particle
			if ( genLowestCost<globalLowestCost ) {
				// only move center if we have improved with this gen
				Network delta = genBest.subtract(globalBest);
				mu = globalBest.add(delta.scale(learningRate));
//				mu = genBest;
				sigma = delta.abs();
				globalBest = genBest;
				globalLowestCost = genLowestCost;
			}
			correct = globalBest.fitness(X, labels);
			System.out.printf("global lowest cost %3.2f, num correct %d %2.2f%%\n",
			                  globalLowestCost, correct, 100*correct/(float) training.size());
		}
	}
}
