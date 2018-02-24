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

    /**
     * 1次元または2次元の行列を文字列化します。
     * @param a 1次元または2次元のINDArray
     * @return INDArrayの文字列表現
     */
    public static String toStringExact(INDArray a) {
        StringBuilder sb = new StringBuilder();
        int rows = a.size(0);
        int cols = a.size(1);
        if (rows != 1) sb.append("[");
        for (int r = 0; r < rows; ++r) {
            if (r > 0) sb.append(", ");
            sb.append("[");
            for (int c = 0; c < cols; ++c) {
                if (c > 0) sb.append(", ");
                sb.append(a.getDouble(r, c));
            }
            sb.append("]");
        }
        if (rows != 1) sb.append("]");
        return sb.toString();
    }

}
