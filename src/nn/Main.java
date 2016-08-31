package nn;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static com.sun.tools.doclint.Entity.mu;
import static com.sun.tools.doclint.Entity.sigma;

public class Main {
	public static final int NUM_PARTICLES = 20;
	public static final int NUM_ITERATIONS = 100;

	static double[][] X;
	static double[] labels;

	public static void main(String[] args) throws IOException {
		List<Image> training = MNISTLoader.loadTrainingImages("/Users/parrt/data/mnist");
		System.out.printf("Loaded %d training images\n", training.size());
//		List<Image> testing = MNISTLoader.loadTestingImages("/Users/parrt/data/mnist");
//		System.out.printf("Loaded %d testing images\n", testing.size());

		training = training.subList(0,1000);
		System.out.printf("Using %d images\n", training.size());

		X = new double[training.size()][];
		labels = new double[training.size()];
		for (int i = 0; i<X.length; i++) {
			X[i] = training.get(i).data;
			labels[i] = training.get(i).label;
		}
		walk(training);
	}

	public static void walk(List<Image> training) {
		double w = 0.729844;
		double c1 = 1.49618, c2 = 1.49618;

		// init all particles
		Particle[] particles = new Particle[NUM_PARTICLES];
		int max = -1;
		int maxi = -0;
		for (int j = 0; j<NUM_PARTICLES; j++) {
			Network net = new Network(784, 15, 10);
			int correct = net.fitness(X, labels);
			particles[j] = new Particle(net,correct);
			if ( correct>max ) {
				max = correct;
				maxi = j;
			}
			System.out.printf(j+": num correct %d %2.2f%%\n", correct, 100*correct/(float)training.size());
		}

		Particle gbest = particles[maxi];
		System.out.println("gbest index is "+maxi+": "+gbest.bestScore);

		for (int i = 0; i<NUM_ITERATIONS; i++) {
			// v(t+1) = w*v(t) + c1*r1*(pBest(t) - x(t)) + c2*r2*(gBest(t) - x(t))
			// x(t+1) = x(t) + v(t+1)
			Network mu = gbest.best;
			double sigma = 1.0;
			for (int j = 0; j<NUM_PARTICLES; j++) {
				Particle p = particles[j];
//				Network net = new Network(mu, sigma, 784, 15, 10);
			}
		}

	}
}
