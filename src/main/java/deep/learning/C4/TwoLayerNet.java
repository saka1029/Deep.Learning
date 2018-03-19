package deep.learning.C4;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

import deep.learning.common.Functions;
import deep.learning.common.INDArrayFunction;
import deep.learning.common.Params;

public class TwoLayerNet {

    public final Params parms;

    /**
     * 重みとバイアスを外部から直接指定するテスト用のコンストラクタです。
     */
    public TwoLayerNet(INDArray W1, INDArray b1, INDArray W2, INDArray b2) {
        this.parms = new Params()
            .put("W1", W1)
            .put("b1", b1)
            .put("W2", W2)
            .put("b2", b2);
    }

    public TwoLayerNet(int input_size, int hidden_size, int output_size) throws Exception {
        this(input_size, hidden_size, output_size, 0.01D);
    }

    public TwoLayerNet(int input_size, int hidden_size, int output_size, double weight_init_std) throws Exception {
        try (Random r = new DefaultRandom()) {
            this.parms = new Params()
                .put("W1", r.nextGaussian(new int[] {input_size, hidden_size}).mul(weight_init_std))
                .put("b1", Nd4j.zeros(hidden_size))
                .put("W2", r.nextGaussian(new int[] {hidden_size, output_size}).mul(weight_init_std))
                .put("b2", Nd4j.zeros(output_size));
        }
    }

    public INDArray predict(INDArray x) {
        INDArray a1 = x.mmul(parms.get("W1")).addRowVector(parms.get("b1"));
        INDArray z1 = Functions.sigmoid(a1);
        INDArray a2 = z1.mmul(parms.get("W2")).addRowVector(parms.get("b2"));
        INDArray y = Functions.softmax(a2);
        return y;
    }

    public double loss(INDArray x, INDArray t) {
        INDArray y = predict(x);
        return Functions.cross_entropy_error(y, t);
    }

    public double accuracy(INDArray x, INDArray t) {
        INDArray y = predict(x);
        // M×N配列のargmaxはM×1行列となります。
        y = Functions.argmax(y);
        t = Functions.argmax(t);
        // y.eq(t)は対応する要素が等しいとき1、そうでないとき0を返します。
        double accuracy = y.eq(t).sumNumber().doubleValue();
        int size = t.size(0);
        // for (int i = 0; i < size; ++i)
        //     if (y.getInt(i) == t.getInt(i))
        //         ++accuracy;
        return accuracy / size;
    }

    public Params numerical_gradient(INDArray x, INDArray t) {
        INDArrayFunction loss_W = W -> loss(x, t);
        return new Params()
            .put("W1", Functions.numerical_gradient(loss_W, parms.get("W1")))
            .put("b1", Functions.numerical_gradient(loss_W, parms.get("b1")))
            .put("W2", Functions.numerical_gradient(loss_W, parms.get("W2")))
            .put("b2", Functions.numerical_gradient(loss_W, parms.get("b2")));
    }

}
