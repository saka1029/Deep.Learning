package deep.learning.test;

import java.io.File;
import java.io.IOException;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import deep.learning.common.Constants;
import deep.learning.common.HistogramImage;

public class TestHistogramImage {

    @Test
    public void testHistogramImage() throws IOException {
        INDArray a = Nd4j.randn(new int[] {1000});
        HistogramImage image = new HistogramImage(500, 500, -4, -10, 4, 100, 30, a);
        if (!Constants.WeightImages.exists())
            Constants.WeightImages.mkdirs();
        image.writeTo(new File(Constants.WeightImages, "TestHistogramImage.png"));
    }

}
