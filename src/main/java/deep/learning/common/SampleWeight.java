package deep.learning.common;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * 「ゼロから作るDeep Learning
 * Pythonで学ぶディープラーニングの理論と実装」
 * で記述された手書き数字認識を行うニューラルネットワークの
 * サンプルウェイトデータを読み込むためのクラスです。
 * 読み込むサンプルウェイトデータは以下の場所にあります。
 * <br>
 * <a href="https://github.com/oreilly-japan/deep-learning-from-scratch/tree/master/ch03">deep-learning-from-scratch/ch03</a>
 * <br>
 * この中の「sample_weight.pkl」がそれです。
 * ただし、これはPythonのマップをシリアライズしたバイナリファイルです。
 * 以下のPythonプログラムを使用して一度テキストに変換する必要があります。
 * <pre><code>
 * import pickle
 * import numpy
 *
 * pkl = "sample_weight.pkl"
 * with open(pkl, "rb") as f:
 * network = pickle.load(f)
 * for k, v in network.items():
 *     print(k, end="")
 *     dim = v.ndim
 *     for d in v.shape:
 *         print("", d, end="")
 *     print()
 *     for e in v.flatten():
 *         print(e)
 * </code></pre>
 * このプログラムは以下のようなテキストを標準出力に書き出します。
 * <pre><code>
 * b2 100
 * -0.014711079
 * -0.07215131
 * -0.0015569247
 * 0.12199665
 * 0.11603302
 * -0.007549459
 * 0.040854506
 * -0.08496164
 * .....
 * </code></pre>
 * これをリダイレクトして適当なテキストファイルに保存し、
 * このプログラムの入力として使います。
 */
public class SampleWeight {

    /**
     * 「ゼロから作るDeep Learning
     * Pythonで学ぶディープラーニングの理論と実装」
     * で記述された手書き数字認識を行うニューラルネットワークの
     * サンプルウェイトデータを読み込みます。
     *
     * @param input sample_weight.pklをテキスト化したファイルを指定します。
     * @return サンプルウェイトデータをMapとして返します。
     * @throws IOException
     */
    public static Map<String, INDArray> read(File input) throws IOException {
        Map<String, INDArray> weights = new HashMap<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(input))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] header = line.split("\\s+");
                INDArray value;
                if (header.length == 2) {
                    int rows = Integer.parseInt(header[1]);
                    value = Nd4j.create(rows);
                    weights.put(header[0], value);
                    for (int r = 0; r < rows; ++r)
                        value.putScalar(r, Float.parseFloat(reader.readLine()));
                } else if (header.length == 3) {
                    int rows = Integer.parseInt(header[1]);
                    int cols = Integer.parseInt(header[2]);
                    value = Nd4j.create(rows, cols);
                    weights.put(header[0], value);
                    for (int r = 0; r < rows; ++r)
                        for (int c = 0; c < cols; ++c)
                            value.putScalar(r, c, Float.parseFloat(reader.readLine()));
                } else
                    throw new IOException("Invalid format: " + line);
            }
        }
        return weights;
    }

}
