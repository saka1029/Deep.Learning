package deep.learning.C5;

import org.nd4j.linalg.api.ndarray.INDArray;

import deep.learning.common.Functions;

public class SoftmaxWithLoss implements LastLayer {

    double loss;
    INDArray y, t;

    @Override
    public double forward(INDArray x, INDArray t) {
        this.t = t;
        this.y = Functions.softmax(x);
        this.loss = Functions.cross_entropy_error(this.y, this.t);
        return this.loss;
    }

    @Override
    public INDArray backward(INDArray x) {
        int batch_size = this.t.size(0);
        return this.y.sub(this.t).div(batch_size);
    }
}
