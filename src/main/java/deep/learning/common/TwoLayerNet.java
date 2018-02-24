package deep.learning.common;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

public class TwoLayerNet {

    public final TwoLayerParams parms;

    public TwoLayerNet(int input_size, int hidden_size, int output_size) throws Exception {
        this(input_size, hidden_size, output_size, 0.01F);
    }

    public TwoLayerNet(int input_size, int hidden_size, int output_size, double weight_init_std) throws Exception {
        try (Random r = new DefaultRandom()) {
            this.parms = new TwoLayerParams(
                r.nextGaussian(new int[] {input_size, hidden_size}).mul(weight_init_std),
                Nd4j.create(hidden_size),
                r.nextGaussian(new int[] {hidden_size, output_size}).mul(weight_init_std),
                Nd4j.create(output_size));
        }
    }

    public INDArray predict(INDArray x) {
        INDArray a1 = x.mmul(parms.W1).addRowVector(parms.b1);
        INDArray z1 = Functions.sigmoid(a1);
        INDArray a2 = z1.mmul(parms.W2).addRowVector(parms.b2);
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
        int accuracy = 0;
        int size = t.size(0);
        for (int i = 0; i < size; ++i)
            if (y.getInt(i) == t.getInt(i))
                ++accuracy;
        return accuracy / size;
    }

    public TwoLayerParams numerical_gradient(INDArray x, INDArray t) {
        INDArrayFunction loss_W = W -> loss(x, t);
        return new TwoLayerParams(
            Functions.numerical_gradient(loss_W, parms.W1),
            Functions.numerical_gradient(loss_W, parms.b1),
            Functions.numerical_gradient(loss_W, parms.W2),
            Functions.numerical_gradient(loss_W, parms.b2));
    }

}
