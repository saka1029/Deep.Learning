package deep.learning.C5;

import static org.junit.Assert.*;

import org.junit.Test;

public class C5_4_単純なレイヤの実装 {

    static class MulLayer {

        private double x, y;

        public double forward(double x, double y) {
            this.x = x;
            this.y = y;
            return x * y;
        }

        // Javaでは多値を返せないので配列で返します。
        public double[] backward(double dout) {
            return new double[] {dout * y, dout * x};
        }
    }

    @Test
    public void C5_4_1_乗算レイヤの実装() {
        double apple = 100;
        double apple_num = 2;
        double tax = 1.1;
        // Layer
        MulLayer mul_apple_layer = new MulLayer();
        MulLayer mul_tax_layer = new MulLayer();
        // forward
        double apple_price = mul_apple_layer.forward(apple, apple_num);
        double price = mul_tax_layer.forward(apple_price, tax);
        assertEquals(220.0, price, 5e-6);
        // backward
        double dprice = 1;
        double[] dapple_price_tax = mul_tax_layer.backward(dprice);
        double[] dapple_num = mul_apple_layer.backward(dapple_price_tax[0]);
        assertEquals(2.2, dapple_num[0], 5e-6);
        assertEquals(110.0, dapple_num[1], 5e-6);
        assertEquals(200.0, dapple_price_tax[1], 5e-6);
    }

    static class AddLayer {

        public double forward(double x, double y) {
            return x + y;
        }

        public double[] backward(double dout) {
            return new double[] {dout, dout};
        }
    }

    @Test
    public void C5_4_2_加算レイヤの実装() {
        double apple = 100;
        double apple_num = 2;
        double orange = 150;
        double orange_num = 3;
        double tax = 1.1;
        // Layer
        MulLayer mul_apple_layer = new MulLayer();
        MulLayer mul_orange_layer = new MulLayer();
        AddLayer add_apple_orange_layer = new AddLayer();
        MulLayer mul_tax_layer = new MulLayer();
        // forward
        double apple_price = mul_apple_layer.forward(apple, apple_num);
        double orange_price = mul_orange_layer.forward(orange, orange_num);
        double all_price = add_apple_orange_layer.forward(apple_price, orange_price);
        double price = mul_tax_layer.forward(all_price, tax);
        // backward
        double dprice = 1;
        double[] dall_price = mul_tax_layer.backward(dprice);
        double[] dapple_dorange_price = add_apple_orange_layer.backward(dall_price[0]);
        double[] dorange = mul_orange_layer.backward(dapple_dorange_price[1]);
        double[] dapple = mul_apple_layer.backward(dapple_dorange_price[0]);
        assertEquals(715.0, price, 5e-6);
        assertEquals(110.0, dapple[1], 5e-6);
        assertEquals(2.2, dapple[0], 5e-6);
        assertEquals(3.3, dorange[0], 5e-6);
        assertEquals(165.0, dorange[1], 5e-6);
        assertEquals(650.0, dall_price[1], 5e-6);
    }

}
