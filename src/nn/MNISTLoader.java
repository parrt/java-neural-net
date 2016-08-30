// From https://raw.githubusercontent.com/Kricket/neural/master/src/main/java/kricket/neural/mnist/Loader.java
package nn;

import com.sun.xml.internal.bind.v2.runtime.unmarshaller.Loader;

import java.io.FileInputStream;
import java.io.IOException;


import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class MNISTLoader {
	public static final int IMAGES_MAGIC = 0x803;
	public static final int LABELS_MAGIC = 0x801;
	public static final String IMAGES_10K_FILE = "t10k-images-idx3-ubyte";
	public static final String LABELS_10K_FILE = "t10k-labels-idx1-ubyte";
	public static final String IMAGES_TRAINING_FILE = "train-images-idx3-ubyte";
	public static final String LABELS_TRAINING_FILE = "train-labels-idx1-ubyte";

	public static List<Image> loadTrainingImages(String dataDir) throws IOException {
		return loadImages(dataDir+'/'+IMAGES_TRAINING_FILE, dataDir+'/'+LABELS_TRAINING_FILE);
	}

	public static List<Image> loadTestingImages(String dataDir) throws IOException {
		return loadImages(dataDir+'/'+IMAGES_10K_FILE, dataDir+'/'+LABELS_10K_FILE);
	}

	public static List<Image> loadImages(String imagesFile, String labelsFile) throws IOException {
		byte[] labels = loadLabels(labelsFile);

		InputStream imgStream = new FileInputStream(imagesFile); //Loader.class.getClassLoader().getResourceAsStream(imagesFile);
		int readMagic = read4(imgStream);
		if(readMagic != IMAGES_MAGIC)
			throw new UnsupportedOperationException("Incorrect magic number: should be " + IMAGES_MAGIC + ", but was " + readMagic);

		int numImages = read4(imgStream);
		if(numImages != labels.length)
			throw new UnsupportedOperationException("We have " + labels.length + " labels, but " + numImages + " images!");

		int numRows = read4(imgStream);
		if(numRows != Image.HEIGHT)
			throw new UnsupportedOperationException("Unsupported image height " + numRows);
		int numCols = read4(imgStream);
		if(numCols != Image.WIDTH)
			throw new UnsupportedOperationException("Unsupported image width " + numCols);

		List<Image> images = new ArrayList<>(numImages);
		final int totalPixels = Image.WIDTH * Image.HEIGHT;

		for(int i=0; i<numImages; i++) {
			double[] image = new double[totalPixels];
			byte[] data = new byte[totalPixels];
			imgStream.read(data);
			for(int d=0; d<totalPixels; d++)
				image[d] = toDouble(data[d]);
			images.add(new Image(image, labels[i]));
		}

		return images;
	}

	public static int toInt(byte b) {
		return ((int) b & 0xFF);
	}

	public static double toDouble(byte b) {
		return toInt(b) / 255.;
	}

	public static byte[] loadLabels(String labelsFile) throws IOException {
		InputStream lblStream = new FileInputStream(labelsFile);//Loader.class.getClassLoader().getResourceAsStream(labelsFile);
		int readMagic = read4(lblStream);
		if(readMagic != LABELS_MAGIC)
			throw new UnsupportedOperationException("Incorrect magic number: should be " + LABELS_MAGIC + ", but was " + readMagic);

		int numLabels = read4(lblStream);
		byte[] labels = new byte[numLabels];
		lblStream.read(labels);

		return labels;
	}

	public static int read4(InputStream stream) throws IOException {
		byte[] byte4 = new byte[4];
		stream.read(byte4);
		int result = (toInt(byte4[0])<<24)
				| (toInt(byte4[1])<<16)
				| (toInt(byte4[2])<<8)
				| toInt(byte4[3]);
		return result;
	}
}
