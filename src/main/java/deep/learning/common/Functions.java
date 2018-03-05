package deep.learning.common;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Functions {

    /**
     * 交差エントロピー誤差(cross entropy error)の実装です。
     *
     * @param y
     * @param t
     * @return
     */
    public static double cross_entropy_error(INDArray y, INDArray t) {
        double delta = 1e-7;
        // Python: return -np.sum(t * np.log(y + delta))
        // return -t.mul(Transforms.log(y.add(delta))).sumNumber().doubleValue();
        // Nd4jのLossFunctionsを使います。
        return LossFunctions.score(t, LossFunctions.LossFunction.MCXENT, y, 0, 0, false);
    }

    /**
     * 勾配(gradient)の実装です。
     *
     * @param f 対象となる関数を指定します。
     * @param x 1次元または2次元の行列を指定します。
     *          3次元以上は処理できないので注意してください。
     * @return xと同じ形の行列として勾配を返します。
     */
    public static INDArray numerical_gradient(INDArrayFunction f, INDArray x) {
        int rows = x.size(0);
        int cols = x.size(1);
        // 理屈はよくわかりませんが、INDArrayの精度が単精度の場合は
        // 1e-3の方が正しく結果を求められるようです。
        double h = 1e-3;
//        double h = 1e-4;
        INDArray grad = Nd4j.create(x.shape());
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                double tmp_val = x.getDouble(r, c);
                x.putScalar(r, c, tmp_val + h);
                double fxh1 = f.apply(x);
                x.putScalar(r, c, tmp_val - h);
                double fxh2 = f.apply(x);
                double g = (fxh1 - fxh2) / (2.0 * h);
//                System.out.printf("fxh1=%f fxh2=%f g=%f%n", fxh1, fxh2, g);
                grad.putScalar(r, c, g);
                x.putScalar(r, c, tmp_val);
            }
        return grad;
    }

    /**
     * ソフトマックス関数の実装です。
     *
     * @param x
     * @return
     */
    public static INDArray softmax(INDArray a) {
        return Transforms.softmax(a);
        // 以下のように実装することもできます。
        // Number c = a.maxNumber();
        // INDArray exp_a = Transforms.exp(a.sub(c));
        // Number sum_exp_a = exp_a.sumNumber();
        // INDArray y = exp_a.div(sum_exp_a);
        // return y;
    }

    /**
     * 各列における最大値のインデックスを求めます。
     * NumPyにおけるagmax関数です。
     *
     * @param a
     * @return m×n行列の入力に対して結果はm×1行列となります。
     */
    public static INDArray argmax(INDArray a) {
        return Nd4j.getExecutioner().exec(new IAMax(a), 1);
    }

    /**
     * sigmoid関数の実装です。
     */
    public static INDArray sigmoid(INDArray x) {
        return Transforms.sigmoid(x);
        // 以下のように実装することもできます。
        // return Transforms.exp(x.neg()).add(1.0).rdiv(1.0);
    }

    /**
     * ReLU関数の実装です。
     */
    public static INDArray relu(INDArray x) {
        return Transforms.relu(x);
    }

    /**
     * 平均を求めます。
     */
    public static double average(INDArray x) {
        // x.length()はすべての要素数を返します。
        return x.sumNumber().doubleValue() / x.length();
    }

}
