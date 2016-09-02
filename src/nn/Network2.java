package nn;

public class Network2 {
	/** Merge biases, weights of network into flat array.
	 *  Layout is biases then weights for layer i
	 */
	public final double[] parameters;
	public final int[] topology;       // {num input units, hidden layers..., num output layer units]

	public Network2(int ...topology) {
		this.topology = topology;
		parameters = new double[sizeOf(topology)];
	}

	public double getBias(int layer, int i) {
		return 0;
	}

	public double getWeight(int layer, int i) {
		return 0;
	}

	public int sizeOf(int[] topology) {
		int n = 0;
		for (int i = 1; i<topology.length; i++) {
			n += topology[i]; // biases
			n += topology[i] * topology[i-1]; // weights = neurons * prevlayerneurons
		}
		return n;
	}

}
