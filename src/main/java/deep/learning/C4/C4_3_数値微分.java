package deep.learning.C4;

import static org.junit.Assert.*;

import java.util.function.DoubleUnaryOperator;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

public class C4_3_数値微分 {

    public static double numerical_diff_bad(DoubleUnaryOperator f, double x) {
        double h = 10e-50;
        return (f.applyAsDouble(x + h) - f.applyAsDouble(x)) / h;
    }

    @Test
    public void C4_3_1_微分() {
        assertEquals(0.0, (float)1e-50, 1e-52);
    }

    public static double numerical_diff(DoubleUnaryOperator f, double x) {
        double h = 1e-4;
        return (f.applyAsDouble(x + h) - f.applyAsDouble(x - h)) / (h * 2);
    }

    public double function_1(double x) {
        return 0.01 * x * x + 0.1 * x;
    }

    public double function_1_diff(double x) {
        return 0.02 * x + 0.1;
    }

    @Test
    public void C4_3_2_数値微分の例() {
        assertEquals(0.200, numerical_diff(this::function_1, 5), 5e-6);
        assertEquals(0.300, numerical_diff(this::function_1, 10), 5e-6);
        assertEquals(0.200, function_1_diff(5), 5e-6);
        assertEquals(0.300, function_1_diff(10), 5e-6);
    }

    public double function_2(INDArray x) {
        double x0 = x.getDouble(0);
        double x1 = x.getDouble(1);
        return x0 * x0 + x1 * x1;
    }

    @Test
    public void C4_3_3_偏微分() {
        DoubleUnaryOperator function_tmp1 = x0 -> x0 * x0 + 4.0 * 4.0;
        assertEquals(6.00, numerical_diff(function_tmp1, 3.0), 5e-6);
        DoubleUnaryOperator function_tmp2 = x1 -> 3.0 * 3.0 + x1 * x1;
        assertEquals(8.00, numerical_diff(function_tmp2, 4.0), 5e-6);
    }

}
