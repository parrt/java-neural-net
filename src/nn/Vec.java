package nn;

import java.util.Arrays;

public class Vec {
	public static double sum(double[] x) {
		return Arrays.stream(x).sum();
	}

	public static double[] abs(double[] x) {
		double[] r = new double[x.length];
		for (int i = 0; i<x.length; i++) {
			r[i] = Math.abs(x[i]);
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
}
