package deep.learning.C3;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import deep.learning.common.Constants;
import deep.learning.common.MNISTImages;
import deep.learning.common.SampleWeight;
import deep.learning.common.Util;

public class C3_6_手書き数字認識 {

    @Test
    public void C3_6_1_MNISTデータセット() throws IOException {
        // MNISTデータセットはMNISTImagesクラスに読み込みます。
        MNISTImages train = new MNISTImages(Constants.TrainImages, Constants.TrainLabels);
        MNISTImages test = new MNISTImages(Constants.TestImages, Constants.TestLabels);
        assertEquals(60000, train.size);
        assertEquals(784, train.imageSize);
        assertEquals(10000, test.size);
        assertEquals(784, test.imageSize);

        // 訓練データの先頭の100イメージをPNGとして出力します。
        if (!Constants.TrainImagesOutput.exists())
            Constants.TrainImagesOutput.mkdirs();
        for (int i = 0; i < 100; ++i) {
            File image = new File(Constants.TrainImagesOutput,
                String.format("%05d-%d.png", i, train.label(i)));
            train.writePngFile(i, image);
        }
    }

    static INDArray normalize(byte[][] images) {
        int imageCount = images.length;
        int imageSize = images[0].length;
        INDArray norm = Nd4j.create(imageCount, imageSize);
        for (int i = 0; i < imageCount; ++i)
            for (int j = 0; j < imageSize; ++j)
                norm.putScalar(i, j, (images[i][j] & 0xff) / 255.0);
        return norm;
    }

    static INDArray predict(Map<String, INDArray> network, INDArray x) {
        INDArray W1 = network.get("W1");
        INDArray W2 = network.get("W2");
        INDArray W3 = network.get("W3");
        INDArray b1 = network.get("b1");
        INDArray b2 = network.get("b2");
        INDArray b3 = network.get("b3");

        // 以下のようにするとバッチ処理でエラーとなります。
        // INDArray a1 = x.mmul(W1).add(b1);
        // x.mmul(W1)の結果が2次元配列なのにb1が1次元であるためです。
        // add(INDArray)は自動的にブロードキャストしません。
        // 以下のように明示的にブロードキャストすることもできます。
        // INDArray a1 = x.mmul(W1).add(b1.broadcast(x.size(0), b1.size(1)));
        INDArray a1 = x.mmul(W1).addRowVector(b1);
        INDArray z1 = Transforms.sigmoid(a1);
        INDArray a2 = z1.mmul(W2).addRowVector(b2);
        INDArray z2 = Transforms.sigmoid(a2);
        INDArray a3 = z2.mmul(W3).addRowVector(b3);
        INDArray y = Transforms.softmax(a3);

        return y;
    }

    @Test
    public void C3_6_2_ニューラルネットワークの推論処理() throws IOException {
        // テスト用のイメージを読み込ます。
        MNISTImages test = new MNISTImages(Constants.TestImages, Constants.TestLabels);
        // サンプルウェイトデータを読み込みます。
        Map<String, INDArray> network = SampleWeight.read(Constants.SampleWeights);
        // イメージを正規化します(0-255 -> 0.0-1.0)
        INDArray x = test.normalizedImages();
        int size = x.size(0);
        int accuracy_cnt = 0;
        for (int i = 0; i < size; ++i) {
            INDArray y = predict(network, x.getRow(i));
            // 最後の引数1は次元を表します。
            INDArray max = Nd4j.getExecutioner().exec(new IAMax(y), 1);
            if (max.getInt(0) == test.label(i))
                ++accuracy_cnt;
        }
//        System.out.printf("Accuracy:%f%n", (double) accuracy_cnt / size);
        assertEquals(10000, size);
        assertEquals(9352, accuracy_cnt);
    }

    @Test
    public void C3_6_3_バッチ処理() throws IOException {
        int batch_size = 100;
        // テスト用のイメージを読み込ます。
        MNISTImages test = new MNISTImages(Constants.TestImages, Constants.TestLabels);
        // サンプルウェイトデータを読み込みます。
        Map<String, INDArray> network = SampleWeight.read(Constants.SampleWeights);
        // イメージを正規化します(0-255 -> 0.0-1.0)
        INDArray x = test.normalizedImages();
        int size = x.size(0);
        int accuracy_cnt = 0;
        for (int i = 0; i < size; i += batch_size) {
            // バッチサイズ分のイメージを取り出してpredict()を呼びます。
            INDArray y = predict(network, x.get(NDArrayIndex.interval(i, i + batch_size)));
            // 最後の引数1は次元を表します。
            INDArray max = Nd4j.getExecutioner().exec(new IAMax(y), 1);
            for (int j = 0; j < batch_size; ++j)
                if (max.getInt(j) == test.label(i + j))
                    ++accuracy_cnt;
        }
//        System.out.printf("Accuracy:%f%n", (double) accuracy_cnt / size);
        assertEquals(10000, size);
        assertEquals(9352, accuracy_cnt);
    }

    @Test
    public void testRange() {
        // NumPyのrange()は以下のように実現できます。
        INDArray a = Nd4j.create(new double[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        // 引数の順序がrange()とは異なる点に注意してください。
        // NumPy: range(start, stop, step)
        // ND4J : NDArrayIndex.interval(int start, int stride, int end);
        assertEquals("[0.00,3.00,6.00,9.00]", Util.string(a.get(NDArrayIndex.interval(0, 3, 10))));
    }

    @Test
    public void testArgmax() {
        // NumPyのargmax()は以下のように実現できます。
        INDArray a = Nd4j.create(new double[][] {{0, 5, 2}, {8, 3, 3}, {6, 5, 4}});
        // 各行の最大インデックスを求めます。
        INDArray rowMax = Nd4j.getExecutioner().exec(new IAMax(a), 1);
        assertEquals("[1.00,0.00,0.00]", Util.string(rowMax));
        // 各列の最大インデックスを求めます。
        INDArray colMax = Nd4j.getExecutioner().exec(new IAMax(a), 0);
        assertEquals("[1.00,0.00,2.00]", Util.string(colMax));
    }

    @Test
    public void testBroadcast() {
        INDArray nd2 = Nd4j.create(new double[]{1, 2, 3});
        INDArray b = nd2.broadcast(new int[]{3, nd2.size(1)});
        String expected = "[[1.00,2.00,3.00],[1.00,2.00,3.00],[1.00,2.00,3.00]]";
        assertEquals(expected, Util.string(b));
    }

    @Test
    public void testSampleWeightRead() throws IOException {
        Map<String, INDArray> w = SampleWeight.read(Constants.SampleWeights);
        assertEquals(784, w.get("W1").size(0));
        assertEquals(50, w.get("W1").size(1));
        assertEquals(50, w.get("W2").size(0));
        assertEquals(100, w.get("W2").size(1));
        assertEquals(100, w.get("W3").size(0));
        assertEquals(10, w.get("W3").size(1));
        assertEquals(50, w.get("b1").size(1));
        assertEquals(100, w.get("b2").size(1));
        assertEquals(10, w.get("b3").size(1));
        assertEquals(-0.007412489, w.get("W1").getDouble(0, 0), 5e-8);
        assertEquals(-0.007904391, w.get("W1").getDouble(0, 1), 5e-8);
        assertEquals(-0.040211476, w.get("W1").getDouble(783, 49), 5e-8);
    }

    @Test
    public void testMNISTTrainImagesWritePNG() throws IOException {
        MNISTImages images = new MNISTImages(Constants.TrainImages, Constants.TrainLabels);
        assertEquals(60000, images.size);
        assertEquals(28, images.rows);
        assertEquals(28, images.columns);
        if (!Constants.TrainImagesOutput.exists())
            Constants.TrainImagesOutput.mkdirs();
        for (int i = 0; i < 100; ++i)
            images.writePngFile(i,
                new File(Constants.TrainImagesOutput,
                    String.format("%05d-%d.png", i, images.label(i))));
    }

    @Test
    public void testMNISTTestImagesWritePNG() throws IOException {
        MNISTImages images = new MNISTImages(Constants.TestImages, Constants.TestLabels);
        assertEquals(10000, images.size);
        assertEquals(28, images.rows);
        assertEquals(28, images.columns);
        if (!Constants.TestImagesOutput.exists())
            Constants.TestImagesOutput.mkdirs();
        for (int i = 0; i < 100; ++i)
            images.writePngFile(i,
                new File(Constants.TestImagesOutput,
                    String.format("%05d-%d.png", i, images.label(i))));
    }
}