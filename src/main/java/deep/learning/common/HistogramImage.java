package deep.learning.common;

import java.awt.Color;

import org.nd4j.linalg.api.ndarray.INDArray;

public class HistogramImage extends GraphImage {

    public HistogramImage(int width, int height,
        double minX, double minY, double maxX, double maxY,
        int divX, INDArray values) {
        super(width, height, minX, minY, maxX, maxY);
        values = values.ravel();    // flatten
        int size = values.length();
        double min = values.minNumber().doubleValue();
        double max = values.maxNumber().doubleValue();
        double step = (maxX - minX) / divX;
        int[] counts = new int[divX + 1];
        for (int i = 0; i < size; ++i) {
            double v = values.getDouble(i);
            int index = (int)((v - min) / step);
            if (index < divX + 1)
                ++counts[index];
        }
        double x = min;
        color(Color.BLUE);
        for (int i = 0; i < divX + 1; ++i, x += step)
            box(x, counts[i], step, counts[i]);
        color(Color.BLACK);
        textInt(String.format("x=(%g, %g) y=(%d, %d)", minX, maxX, (int)minY, (int)maxY), 0, 10);
    }

}
