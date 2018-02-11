package deep.learning.C3;

import static org.junit.Assert.*;

import java.util.function.DoubleFunction;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import test.deep.learning.Util;

public class C3__2_活性化関数 {

    public static double step_function(double x) {
        if (x > 0)
            return 1.0;
        else
            return 0.0;
    }

    public static <T extends Number> INDArray map(INDArray x, DoubleFunction<T> func) {
        int size = x.length();
        INDArray result = Nd4j.create(size);
        for (int i = 0; i < size; ++i)
            result.put(0, i, func.apply(x.getDouble(i)));
        return result;
    }

    public static INDArray step_function(INDArray x) {
        return map(x, d -> d > 0.0 ? 1 : 0);
    }

    @Test
    public void C3_2_2_ステップ関数の実装() {
        INDArray x = Nd4j.create(new double[] {-1.0, 1.0, 2.0});
        assertEquals("[-1.00,1.00,2.00]", Util.string(x));
        assertEquals("[0.00,1.00,1.00]", Util.string(step_function(x)));
    }

    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public static INDArray sigmoid(INDArray x) {
        // Javaでは演算子のオーバーロードができないので
        // メソッド呼び出しで記述します。
        return Transforms.exp(x.neg()).add(1.0).rdiv(1.0);
        // あるいは以下のように実装することもできます。
        // return map(x, d -> sigmoid(d));
    }

    @Test
    public void C3_2_4_シグモイド関数の実装() {
        INDArray x = Nd4j.create(new double[] {-1.0, 1.0, 2.0});
        assertEquals("[0.27,0.73,0.88]", Util.string(sigmoid(x)));
        // ND4Jにはorg.nd4j.linalg.ops.transforms.Transformsクラスにsigmoid関数が用意されています。
        assertEquals("[0.27,0.73,0.88]", Util.string(Transforms.sigmoid(x)));
        INDArray t = Nd4j.create(new double[] {1.0, 2.0, 3.0});
        assertEquals("[2.00,3.00,4.00]", Util.string(t.add(1.0)));
        assertEquals("[1.00,0.50,0.33]", Util.string(t.rdiv(1.0)));
    }

    public static INDArray relu(INDArray x) {
        return map(x, d -> Math.max(0.0, d));
    }

    @Test
    public void C3_2_7_ReLU関数() {
        INDArray x = Nd4j.create(new double[] {-4, -2, 0, 2, 4});
        assertEquals("[0.00,0.00,0.00,2.00,4.00]", Util.string(relu(x)));
        // ND4Jにはorg.nd4j.linalg.ops.transforms.Transformsクラスにrelu関数が用意されています。
        assertEquals("[0.00,0.00,0.00,2.00,4.00]", Util.string(Transforms.relu(x)));
    }

}
