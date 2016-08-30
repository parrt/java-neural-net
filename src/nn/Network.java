package nn;

import lib.Matrix;

public class Network {
	public double[][] biases;
	public double[][][] weights;
	public int[] topology;

	public Network(int ...topology) {
		this.topology = topology;
		biases = new double[topology.length-1][];
		weights = new double[topology.length-1][][];
		for (int i = 1; i<topology.length; i++) {
			biases[i] = new double[topology[i]];
			weights[i] = new double[topology[i]][topology[i-1]];
		}
	}

	public double[] feedforward(double[] activations) {
		for (int i = 1; i<topology.length; i++) {
			double[] b = biases[i];
			double[][] w = weights[i];
			activations = reLU( Matrix.multiply(w, b) );
		}
		return activations;
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
}
