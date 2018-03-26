package deep.learning.common;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * common/layers.pyのDropoutクラスのJava版です。
 */
public class Dropout implements BatchNormLayer {

    final double dropout_ratio;
    INDArray mask = null;

    public Dropout(double dropout_ratio/*=0.5*/) {
        this.dropout_ratio = dropout_ratio;
    }

    @Override
    public INDArray forward(INDArray x, boolean train_flg) {
        if (train_flg) {
            mask = Nd4j.rand(x.shape()).gt(dropout_ratio);
            return x.mul(mask);
        } else
            return x.mul(1.0 - dropout_ratio);
    }

    @Override
    public INDArray backward(INDArray dout) {
        return dout.mul(mask);
    }
}
