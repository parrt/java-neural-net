package nn;

import lib.StdArrayIO;

import java.util.Arrays;

public class Vec {
	public static double sum(double[] x) {
		return Arrays.stream(x).sum();
	}

	public static double norm(double[] x) {
		double s = 0.0;
		for (int i = 0; i<x.length; i++) {
			s += x[i]*x[i];
		}
		double r = Math.sqrt(s);
		return r;
	}

	public static double[] abs(double[] x) {
		double[] r = new double[x.length];
		for (int i = 0; i<x.length; i++) {
			r[i] = Math.abs(x[i]);
		}
		return r;
	}

	public static double[] exp(double[] x) {
		double[] r = new double[x.length];
		for (int i = 0; i<x.length; i++) {
			r[i] = Math.exp(x[i]);
		}
		return r;
	}

	public static double[] ones(int n) {
		double[] r = new double[n];
		for (int i = 0; i<n; i++) {
			r[i] = 1.0;
		}
		return r;
	}

	public static double[] scale(double[] x, double v) {
		double[] r = new double[x.length];
		for (int i = 0; i<x.length; i++) {
			r[i] = x[i] * v;
		}
		return r;
	}

	public static double[] add(double[] x, double[] y) {
		if ( x.length!=y.length ) {
			throw new IllegalArgumentException("diff vec lens");
		}
		double[] r = new double[x.length];
		for (int i = 0; i<x.length; i++) {
			r[i] = x[i] + y[i];
		}
		return r;
	}

	public static double[] subtract(double[] x, double[] y) {
		if ( x.length!=y.length ) {
			throw new IllegalArgumentException("diff vec lens");
		}
		double[] r = new double[x.length];
		for (int i = 0; i<x.length; i++) {
			r[i] = x[i] - y[i];
		}
		return r;
	}

	public static void main(String[] args) {
		/*
		 31.20000   3.40000  55.00000
		-15.60000   1.70000 -27.50000
		  0.00000  13.40000 -50.00000
		-62.40000  -6.60000  50.00000
		 */
		StdArrayIO.print(abs(new double[]{-31.2, 3.4, -55}));
		StdArrayIO.print(scale(new double[]{-31.2, 3.4, -55}, 0.5));
		StdArrayIO.print(add(new double[]{-31.2, 3.4, -55},
		                     new double[]{31.2, 10, 5}));
		StdArrayIO.print(subtract(new double[]{-31.2, 3.4, 55},
		                          new double[]{31.2, 10, 5}));
	}
}
