package deep.learning.C5;

import org.nd4j.linalg.api.ndarray.INDArray;

import deep.learning.common.Functions;

public class Sigmoid implements Layer {

    INDArray out;

    @Override
    public INDArray forward(INDArray x) {
        out = Functions.sigmoid(x);
        // あるいは
        // out = Transforms.exp(x.neg()).add(1.0).rdiv(1.0);
        return out;
    }

    @Override
    public INDArray backward(INDArray dout) {
        return dout.mul(out.rsub(1.0)).mul(out);
    }
}
