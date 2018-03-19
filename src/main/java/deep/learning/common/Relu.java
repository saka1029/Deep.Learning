package deep.learning.common;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Relu implements Layer {

    public INDArray mask;

    @Override
    public INDArray forward(INDArray x) {
        // 要素の値＞0.0の時は1、それ以外の時は0をmaskに格納します。
        // "gt"は"greater than"の意味です。
        this.mask = x.gt(0.0);
        return Functions.relu(x);
    }

    @Override
    public INDArray backward(INDArray dout) {
        return dout.mul(mask);
    }

}
