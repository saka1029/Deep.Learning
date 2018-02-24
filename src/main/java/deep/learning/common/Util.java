package deep.learning.common;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Util {

    public static INDArray identity(INDArray array) {
        return Transforms.identity(array);
    }

    public static INDArray softmax(INDArray array) {
        return Nd4j.getExecutioner().execAndReturn(new SoftMax(array.dup()));
    }

    public static INDArray sigmoid(INDArray array) {
        return Transforms.sigmoid(array);
    }

    public static INDArray exp(INDArray array) {
        return Transforms.exp(array);
    }

    public static float max(INDArray array) {
        return array.maxNumber().floatValue();
    }

    public static float min(INDArray array) {
        return array.minNumber().floatValue();
    }

    public static double sum(INDArray array) {
        return array.sumNumber().doubleValue();
    }

    public static String string(INDArray a) {
        return a.toString().replaceAll("\\s", "");
    }

}
