package nn;

import lib.Matrix;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public class Network {
	public final double[][] biases;    // biases[layer][neuron]
	public final double[][][] weights; // weights[layer][neuron][neuron-from-prev-layer]
	public final int[] topology;       // {num input units, hidden layers..., num output layer units]

	public Network(int ...topology) {
		this.topology = topology;
		biases = new double[topology.length-1][];
		weights = new double[topology.length-1][][];
		for (int i = 1; i<topology.length; i++) {
			biases[i-1] = new double[topology[i]];
			weights[i-1] = new double[topology[i]][topology[i-1]];
		}
	}

	public Network(Network mu, Network sigma, int ...topology) { // init with N(mu,sigma)
		this.topology = topology;
		biases = new double[topology.length-1][];
		weights = new double[topology.length-1][][];
		// init parameters with N(mu,sigma)
		for (int i = 1; i<topology.length; i++) {
			biases[i-1] = Matrix.randomGaussian(topology[i], mu.biases[i-1], sigma.biases[i-1]);
			weights[i-1] = Matrix.randomGaussian(topology[i], topology[i-1], mu.weights[i-1], sigma.weights[i-1]);
		}
	}

	public Network(double mu, double sigma, int ...topology) { // init with U(mu, sigma)
		this.topology = topology;
		biases = new double[topology.length-1][];
		weights = new double[topology.length-1][][];
		for (int i = 1; i<topology.length; i++) {
			biases[i-1] = Matrix.randomGaussian(topology[i], mu, sigma);
			weights[i-1] = Matrix.randomGaussian(topology[i], topology[i-1], mu, sigma);
		}
	}

	public static Network ones(int ...topology) {
		Network n = new Network(topology);
		for (int i = 1; i<topology.length; i++) {
			n.biases[i-1] = Vec.ones(topology[i]);
			n.weights[i-1] = Matrix.ones(topology[i], topology[i-1]);
		}
		return n;
	}

	public int size() {
		int n = 0;
		for (int i = 1; i<topology.length; i++) {
			n += topology[i]; // biases
			n += topology[i] * topology[i-1]; // weights
		}
		return n;
	}

	/** Pack all data together into a linear vector */
	public double[] asVector() {
		double[] v = new double[size()];
		int k = 0;
		for (int i = 0; i<topology.length-1; i++) {
			for (int j = 0; j<biases[i].length; j++) {
				v[k++] = biases[i][j];
			}
			for (int j = 0; j<weights[i].length; j++) {
				for (int jj = 0; jj<weights[i][j].length; jj++) {
					v[k++] = weights[i][j][jj];
				}
			}
		}
		return v;
	}

	public Network finiteDifference(double h, double[][] X, double[][] onehots) {
		Network costDiffs = new Network(this.topology);
		for (int i = 0; i<topology.length-1; i++) {
			for (int j = 0; j<biases[i].length; j++) {
				biases[i][j] -= h;
				double cleft = cost(X, onehots);
				biases[i][j] += h;
				biases[i][j] += h;
				double cright = cost(X, onehots);
				biases[i][j] -= h;
				costDiffs.biases[i][j] = (cright - cleft)/(2*h);
			}
			for (int j = 0; j<weights[i].length; j++) {
				for (int k = 0; k<weights[i][j].length; k++) {
					weights[i][j][k] -= h;
					double cleft = cost(X, onehots);
					weights[i][j][k] += h;
					weights[i][j][k] += h;
					double cright = cost(X, onehots);
					weights[i][j][k] -= h;
					costDiffs.weights[i][j][k] = (cright - cleft)/(2*h);
				}
			}
		}
		return costDiffs;
	}

	public Network abs() {
		Network r = new Network(this.topology);
		for (int i = 0; i<topology.length-1; i++) {
			r.biases[i] = Vec.abs(this.biases[i]);
			r.weights[i] = Matrix.abs(this.weights[i]);
		}
		return r;
	}

	public Network scale(double v) {
		Network r = new Network(this.topology);
		for (int i = 0; i<topology.length-1; i++) {
			r.biases[i] = Vec.scale(this.biases[i], v);
			r.weights[i] = Matrix.multiply(this.weights[i], v);
		}
		return r;
	}

	public Network add(Network other) {
		Network r = new Network(this.topology);
		for (int i = 0; i<topology.length-1; i++) {
			r.biases[i] = Vec.add(this.biases[i], other.biases[i]);
			r.weights[i] = Matrix.add(this.weights[i], other.weights[i]);
		}
		return r;
	}

	public Network subtract(Network other) {
		Network r = new Network(this.topology);
		for (int i = 0; i<topology.length-1; i++) {
			r.biases[i] = Vec.subtract(this.biases[i], other.biases[i]);
			r.weights[i] = Matrix.subtract(this.weights[i], other.weights[i]);
		}
		return r;
	}

	public double[] feedforward(double[] activations) {
		for (int layer = 0; layer<topology.length-1; layer++) {
			double[][] w = weights[layer];
			double[] wapplied = Matrix.multiply(w, activations);
			double[] output = Vec.add(wapplied, biases[layer]);
			activations = reLU(output);
		}
		return activations;
	}

	public int fitness(double[][] X, int[] labels) {
		int correct = 0;
		for (int i = 0; i<X.length; i++) {
			double[] x = X[i];
			double[] output = feedforward(x);
			double[] predictions = softmax(output);
			int predicted = argmax(predictions);
			if ( predicted==labels[i] ) {
				correct++;
			}
		}
		return correct;
	}

	public double cost(double[][] X, double[][] onehots) {
		double c = 0.0;
		for (int i = 0; i<X.length; i++) {
			double[] x = X[i];
			double[] output = feedforward(x);
			double[] predictions = softmax(output);
			double[] diff = Vec.subtract(predictions, onehots[i]);
			double norm = Vec.norm(diff);
			c += norm * norm;
		}
		return c / (2 * X.length); // average cost across X.length exemplars
	}

	public double reLU(double v) {
		return Math.max(0, v);
	}

	public double[] reLU(double[] v) {
		double[] re = new double[v.length];
		for (int i = 0; i<v.length; i++) {
			re[i] = reLU(v[i]);
		}
		return re;
	}

	public int argmax(double[] x) {
		double max = Double.MIN_VALUE;
		int maxi = -1;
		for (int i = 0; i<x.length; i++) {
			if ( x[i]>max ) {
				max = x[i];
				maxi = i;
			}
		}
		return maxi;
	}

	public double[] softmax(double[] x) {
		double[] r = new double[x.length];
		double s = Vec.sum(Vec.exp(x));
		for (int i = 0; i<x.length; i++) {
			r[i] = Math.exp(x[i]) / s;
			if ( Double.isNaN(r[i]) ) {
//				r[i] = 1.0;
//				System.out.println("NAN!!!!!!!!!!!");
			}
		}
		return r;
	}

	public static <T, R> List<R> map(T[] data, Function<T, R> getter) {
		List<R> output = new ArrayList<>();
		if ( data!=null ) for (T x : data) {
			output.add(getter.apply(x));
		}
		return output;
	}

	@Override
	public String toString() {
		StringBuilder buf = new StringBuilder();
		buf.append("Biases:\n");
		for (int i = 0; i<biases.length; i++) {
			for (int j = 0; j<biases[i].length; j++) {
				buf.append(String.format(" %3.2f", biases[i][j]));
			}
			buf.append("\n");
		}
		buf.append("\nWeights:\n");
		for (int i = 0; i < weights.length; i++) {
			buf.append("Layer "+(i+1)+":\n");
			for (int j = 0; j < weights[i].length; j++) {
				for (int k = 0; k<weights[i][j].length; k++) {
					buf.append(String.format(" %3.3f", weights[i][j][k]));
				}
				buf.append("\n");
			}
			buf.append("\n");
		}
		return buf.toString();
	}
}
