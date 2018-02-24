package deep.learning.common;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.Closeable;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * 「ゼロから作るDeep Learning
 * Pythonで学ぶディープラーニングの理論と実装」
 * で記述されたMNISTの手書き数字認識用イメージのクラスです。
 * MNISTのデータおよびファイルフォーマットの説明は
 * <a href="http://yann.lecun.com/exdb/mnist/">THE MNIST DATABASE</a>
 * にあります。
 * このクラスは訓練データの読み込み
 * またはテストデータの読み込みで使用することができます。
 */
public class MNISTImages {

    /**
     * イメージの数です。
     */
    public final int size;
    /**
     * １イメージあたりの行数です。
     */
    public final int rows;
    /**
     * １イメージあたりの列数です。
     */
    public final int columns;
    /**
     * イメージあたりのピクセル数です。(rows * columns)
     */
    public final int imageSize;
    /**
     * イメージ(複数)です。
     * 各イメージはフラット化して格納します。
     */
    private final byte[][] images;
    /**
     * ラベル(複数)です。
     * 各要素には0x00から0x09の値が格納されます。
     */
    private final byte[] labels;

    /**
     * 指定されたイメージファイル、ラベルファイルを読み込みます。
     *
     * @param image イメージファイルを指定します。
     * @param label ラベルファイルを指定します。
     * @throws IOException
     */
    public MNISTImages(File image, File label) throws IOException {
        try (DataInputStream in = new DataInputStream(new FileInputStream(image))) {
            int header = in.readInt();
            if (header != 0x00000803)
                throw new IOException("Invalid image header");
            size = in.readInt();
            rows = in.readInt();
            columns = in.readInt();
            imageSize = rows * columns;
            images = new byte[size][imageSize];
            for (int i = 0; i < size; ++i)
                in.readFully(images[i]);
        }
        try (DataInputStream in = new DataInputStream(new FileInputStream(label))) {
            int header = in.readInt();
            if (header != 0x00000801)
                throw new IOException("Invalid label header");
            if (in.readInt() != size)
                throw new IOException("Invalid label size");
            labels = new byte[size];
            in.readFully(labels);
        }
    }

    /**
     * フラット化され正規化されたイメージを返します。
     * @return size * imageSizeの行列を返します。
     *         各要素は0.0から1.0の値に正規化されています。
     */
    public INDArray normalizedImages() {
        INDArray norm = Nd4j.create(size, imageSize);
        for (int i = 0; i < size; ++i)
            for (int j = 0; j < imageSize; ++j)
                norm.putScalar(i, j, (images[i][j] & 0xff) / 255.0);
        return norm;
    }

    /**
     * 指定された位置のラベル値を返します。
     * @param index 取得する位置を指定します。
     * @return 0から9の範囲でラベル値を返します。
     */
    public int label(int index) {
        return labels[index];
    }

    /**
     * すべてのラベル値を返します。
     * @return size列のベクトルを返します。
     *         各要素の値は0から9のいずれかです。
     */
    public INDArray labels() {
        INDArray array = Nd4j.create(size);
        for (int i = 0; i < size; ++i)
            array.putScalar(i, labels[i]);
        return array;
    }

    /**
     * one-hot表現としてのラベルを返します。
     * @return size * 10の行列を返します。
     *         各要素の値は1または0です。
     *         ラベル値が3の場合返される行は
     *         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]です。
     */
    public INDArray oneHotLabels() {
        INDArray oneHot = Nd4j.create(size, 10);
        for (int i = 0; i < size; ++i)
            oneHot.putScalar(i, labels[i], 1);
        return oneHot;
    }

    /**
     * 指定したインデックスのイメージをPNGファイルとして出力します。
     *
     * @param index 出力するイメージのインデックスを指定します。
     * @param path 出力先のファイルを指定します。
     * @throws IOException
     */
    public void writePngFile(int index, File path) throws IOException {
        BufferedImage image = new BufferedImage(columns, rows, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = image.createGraphics();
        try (Closeable cl = () -> g.dispose()) {
            byte[] imageBytes = images[index];
            for (int i = 0, r = 0; r < rows; ++r) {
                for (int c = 0; c < columns; ++c) {
                    int b = imageBytes[i++] & 0xff;
                    g.setColor(new Color(b, b, b));
                    g.fillRect(c, r, 1, 1);
                }
            }
        }
        ImageIO.write(image, "png", path);
    }

    public static void writePngFile(INDArray image, int columns, int rows, File path) throws IOException {
        BufferedImage bi = new BufferedImage(columns, rows, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = bi.createGraphics();
        try (Closeable cl = () -> g.dispose()) {
            for (int i = 0, r = 0; r < rows; ++r) {
                for (int c = 0; c < columns; ++c) {
                    float b = image.getFloat(i++);
                    g.setColor(new Color(b, b, b));
                    g.fillRect(c, r, 1, 1);
                }
            }
        }
        ImageIO.write(bi, "png", path);

    }

}
