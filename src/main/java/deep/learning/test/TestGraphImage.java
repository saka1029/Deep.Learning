package deep.learning.test;

import java.awt.Color;
import java.io.File;
import java.io.IOException;

import org.junit.Test;

import deep.learning.common.GraphImage;

public class TestGraphImage {

    @Test
    public void testGraphImage() throws IOException {
        try (GraphImage g = new GraphImage(400, 400, -2, -2, 2, 2)) {
            g.color(Color.BLACK);
            g.line(0, 0, 1, 1);
            g.color(Color.RED);
            g.plot(0, 0);
            g.plot(1, 1);
            g.plot(-1, 1);
            g.plot(1, -1);
            g.plot(-1, -1);
            g.writeTo(new File("output/images/TestGraphImage.png"));
        }
    }

}
