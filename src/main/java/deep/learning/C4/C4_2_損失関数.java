package deep.learning.C4;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;

import deep.learning.common.Constants;
import deep.learning.common.MNISTImages;

public class C4_2_損失関数 {

    /**
     * ２乗和誤差関数（自身の転置行列との内積で計算）
     */
    public static double mean_squared_error(INDArray y, INDArray t) {
        INDArray diff = y.sub(t);
        return 0.5 * diff.mmul(diff.transpose()).getDouble(0);
    }

    /**
     * ２乗和誤差関数（INDArray.squaredDistance(INDArray)を使用）
     */
    public static double mean_squared_error2(INDArray y, INDArray t) {
        return 0.5 * (double)y.squaredDistance(t);
    }

    @Test
    public void C4_2_1_２乗和誤差() {
        INDArray t = Nd4j.create(new double[] {0, 0, 1, 0, 0, 0, 0, 0, 0, 0});
        INDArray y = Nd4j.create(new double[] {0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0});
        assertEquals(0.097500000000000031, mean_squared_error(y, t), 5e-6);
        assertEquals(0.097500000000000031, mean_squared_error2(y, t), 5e-6);
        // LossFunctions.LossFunction.MSEを使っても実現できます。
        assertEquals(0.097500000000000031, LossFunctions.score(t, LossFunctions.LossFunction.MSE, y, 0, 0, false), 5e-6);
        y = Nd4j.create(new double[] {0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0});
        assertEquals(0.59750000000000003, mean_squared_error(y, t), 5e-6);
        assertEquals(0.59750000000000003, mean_squared_error2(y, t), 5e-6);
        assertEquals(0.59750000000000003, LossFunctions.score(t, LossFunctions.LossFunction.MSE, y, 0, 0, false), 5e-6);
    }

    public static double cross_entropy_error(INDArray y, INDArray t) {
        double delta = 1e-7;
        // Python: return -np.sum(t * np.log(y + delta))
        return -t.mul(Transforms.log(y.add(delta))).sumNumber().doubleValue();
    }

    @Test
    public void C4_2_2_交差エントロピー誤差() {
        INDArray t = Nd4j.create(new double[] {0, 0, 1, 0, 0, 0, 0, 0, 0, 0});
        INDArray y = Nd4j.create(new double[] {0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0});
        assertEquals(0.51082545709933802, cross_entropy_error(y, t), 5e-6);
        // LossFunctionsを使って実現することもできます。
         assertEquals(0.51082545709933802, LossFunctions.score(t, LossFunctions.LossFunction.MCXENT, y, 0, 0, false), 5e-6);
        y = Nd4j.create(new double[] {0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0});
        assertEquals(2.3025840929945458, cross_entropy_error(y, t), 5e-6);
    }

    @Test
    public void C4_2_3_ミニバッチ学習() throws IOException {
        // MNISTデータセットを読み込みます。
        MNISTImages train = new MNISTImages(Constants.TrainImages, Constants.TrainLabels);
        assertArrayEquals(new int[] {60000, 784}, train.normalizedImages().shape());
        assertArrayEquals(new int[] {60000, 10}, train.oneHotLabels().shape());
        // ランダムに10枚のイメージを抽出します。
        // 一度DataSetにイメージとラベルを格納し、サンプルとして指定枚数分を取り出します。
        DataSet ds = new DataSet(train.normalizedImages(), train.oneHotLabels());
        DataSet sample = ds.sample(10);
        assertArrayEquals(new int[] {10, 784}, sample.getFeatureMatrix().shape());
        assertArrayEquals(new int[] {10, 10}, sample.getLabels().shape());
        // 取得されたサンプルのイメージとラベル値の対応があっていることを確認するために
        // サンプルのイメージをPNGファイルとして書き出します。
        // one-hot形式のラベルから元のラベル値へ変換します。（各行の最大値のインデックスを求めます）
        INDArray indexMax = Nd4j.getExecutioner().exec(new IAMax(sample.getLabels()), 1);
        if (!Constants.SampleImagesOutput.exists())
            Constants.SampleImagesOutput.mkdirs();
        for (int i = 0; i < 10; ++i) {
            // ファイル名は"(連番)-(ラベル値).png"となります。
            File f = new File(Constants.SampleImagesOutput,
                String.format("%05d-%d.png",
                    i, indexMax.getInt(i)));
            MNISTImages.writePngFile(sample.getFeatures().getRow(i), train.rows, train.columns, f);
        }
    }

    /**
     * 交差エントロピー誤差を求めます。（バッチ対応版）
     * ND4Jの場合1次元配列は1行N列の行列なので、
     * yが1次元であっても2次元であっても同様に処理できます。
     */
    public static double cross_entropy_error2(INDArray y, INDArray t) {
        int batch_size = y.size(0);
        return -t.mul(Transforms.log(y.add(1e-7))).sumNumber().doubleValue() / batch_size;
    }

    @Test
    public void C4_2_4_バッチ対応版交差エントロピー誤差の実装() {
        // 単一データの場合
        INDArray t = Nd4j.create(new double[] {0, 0, 1, 0, 0, 0, 0, 0, 0, 0});
        INDArray y = Nd4j.create(new double[] {0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0});
        assertEquals(0.51082545709933802, cross_entropy_error2(y, t), 5e-6);
        // バッチサイズ=2の場合（同一データが2件）
        t = Nd4j.create(new double[][] {
            {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 1, 0, 0, 0, 0, 0, 0, 0}});
        y = Nd4j.create(new double[][] {
            {0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0},
            {0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0}});
        assertEquals(0.51082545709933802, cross_entropy_error2(y, t), 5e-6);
        // todo: one-hot表現でない場合の交差エントロピー誤差の実装
    }

}
