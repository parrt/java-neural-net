package nn;

public class Image {
	public static final int MNIST_WIDTH = 28, MNIST_HEIGHT = 28;

	public double[] data;
	public int label;

	public Image(double[] data, int label) {
		this.data = data;
		this.label = label;
	}
}
