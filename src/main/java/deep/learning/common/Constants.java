package deep.learning.common;

import java.io.File;

public class Constants {

    public static File MNIST = new File("input/mnist");
    public static File IMAGES = new File("output/images");
    public static File WEIGHTS = new File("output/weights");

    public static File TrainImages = new File(MNIST, "train-images.idx3-ubyte");
    public static File TrainLabels = new File(MNIST, "train-labels.idx1-ubyte");
    public static File TestImages = new File(MNIST, "t10k-images.idx3-ubyte");
    public static File TestLabels = new File(MNIST, "t10k-labels.idx1-ubyte");
    public static File SampleWeights = new File(MNIST, "sample_weight.txt");
    public static File TrainImagesOutput = new File(IMAGES, "train");
    public static File TestImagesOutput = new File(IMAGES, "test");
    public static File SampleImagesOutput = new File(IMAGES, "sample");

}
