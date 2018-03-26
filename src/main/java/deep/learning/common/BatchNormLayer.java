package deep.learning.common;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface BatchNormLayer extends Layer {

    public default INDArray forward(INDArray x) {
        throw new IllegalAccessError();
    }

    INDArray forward(INDArray x, boolean train_flg);

}
