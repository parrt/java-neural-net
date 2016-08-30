package nn;

import java.io.IOException;
import java.util.List;

public class Main {
	public static void main(String[] args) throws IOException {
		List<Image> training = MNISTLoader.loadTrainingImages("/Users/parrt/data/mnist");
		List<Image> testing = MNISTLoader.loadTestingImages("/Users/parrt/data/mnist");
		Network net = new Network(784, 15, 10);
		System.out.println(net);
//		net.fitness(X, labels);
	}
}
