package nn;

import lib.Matrix;
import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

import static java.util.Arrays.stream;
import static javafx.scene.input.KeyCode.R;
import static org.apache.commons.lang3.ArrayUtils.toArray;
import static sun.management.snmp.jvminstr.JvmThreadInstanceEntryImpl.ThreadStateMap.Byte1.other;

public class Network {
	public double[][] biases;    // biases[layer][neuron]
	public double[][][] weights; // weights[layer][neuron][neuron-from-prev-layer]
	public int[] topology;       // {num input units, hidden layers..., num output layer units]

	public Network() {
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

	public Network(int ...topology) { // init with U(0,1)
		this.topology = topology;
		biases = new double[topology.length-1][];
		weights = new double[topology.length-1][][];
		for (int i = 1; i<topology.length; i++) {
			biases[i-1] = Matrix.random(topology[i]);
			weights[i-1] = Matrix.random(topology[i], topology[i-1]);
		}
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
		for (int i = 1; i<topology.length; i++) {
			for (int j = 0; j<biases[i-1].length; j++) {
				v[k++] = biases[i-1][j];
			}
			for (int j = 0; j<weights[i-1].length; j++) {
				for (int jj = 0; jj<weights[i-1][j].length; jj++) {
					v[k++] = weights[i-1][j][jj];
				}
			}
		}
		return v;
	}

	public Network abs() {
		Network r = new Network();
		r.topology = this.topology;
		for (int i = 1; i<topology.length; i++) {
			r.biases[i] = Vec.abs(this.biases[i]);
			r.weights[i] = Matrix.abs(this.weights[i]);
		}
		return r;
	}

	public Network scale(double v) {
		Network r = new Network();
		r.topology = this.topology;
		for (int i = 1; i<topology.length; i++) {
			r.biases[i] = Vec.scale(this.biases[i], v);
			r.weights[i] = Matrix.multiply(this.weights[i], v);
		}
		return r;
	}

	public Network add(Network other) {
		Network r = new Network();
		r.topology = this.topology;
		for (int i = 1; i<topology.length; i++) {
			r.biases[i] = Vec.add(this.biases[i], other.biases[i]);
			r.weights[i] = Matrix.add(this.weights[i], other.weights[i]);
		}
		return r;
	}

	public Network subtract(Network other) {
		Network r = new Network();
		r.topology = this.topology;
		for (int i = 1; i<topology.length; i++) {
			r.biases[i] = Vec.subtract(this.biases[i], other.biases[i]);
			r.weights[i] = Matrix.subtract(this.weights[i], other.weights[i]);
		}
		return r;
	}

	public double[] feedforward(double[] activations) {
		for (int i = 0; i<weights.length; i++) {
			double[][] w = weights[i];
			double[] output = Vec.add(Matrix.multiply(w, activations), biases[i]);
			activations = reLU(output);
		}
		return activations;
	}

	public int fitness(double[][] X, double[] labels) {
		int correct = 0;
		for (int i = 0; i<X.length; i++) {
			double[] x = X[i];
			double y = labels[i];
			double[] y_ = feedforward(x);
			int predicted = argmax(softmax(y_));
			if ( predicted==y ) {
				correct++;
			}
		}
		return correct;
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

	public double[] softmax(double[] data) {
		double sum = stream(data).map(x -> Math.exp(x)).sum();
		return stream(data).map(x -> x / sum).toArray();
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
					buf.append(String.format(" %3.2f", weights[i][j][k]));
				}
				buf.append("\n");
			}
			buf.append("\n");
		}
		return buf.toString();
	}
}
