package deep.learning.database;

import java.awt.*;
import java.awt.image.*;
import java.io.*;

import javax.imageio.*;

public class MNISTImages {

    public final int size, rows, columns;
    public final byte[][] images;
    public final byte[] labels;

    public MNISTImages(File image, File label) throws IOException {
        try (DataInputStream in = new DataInputStream(new FileInputStream(image))) {
            int header = in.readInt();
            if (header != 0x00000803)
                throw new IOException("Invalid image header");
            size = in.readInt();
            rows = in.readInt();
            columns = in.readInt();
            int imageSize = rows * columns;
            images = new byte[size][imageSize];
            for (int i = 0; i < size; ++i)
                in.readFully(images[i]);
        }
        try (DataInputStream in = new DataInputStream(new FileInputStream(label))) {
            int header = in.readInt();
            if (header != 0x00000801)
                throw new IOException("Invalid label header");
            if (in.readInt() != size)
                throw new IOException("Invalid label size");
            labels = new byte[size];
            in.readFully(labels);
        }
    }

    public void writePngFile(int index, File path) throws IOException {
        BufferedImage image = new BufferedImage(columns, rows, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = image.createGraphics();
        try (Closeable cl = () -> g.dispose()) {
            byte[] imageBytes = images[index];
            for (int i = 0, r = 0; r < rows; ++r) {
                for (int c = 0; c < columns; ++c) {
                    int b = imageBytes[i++] & 0xff;
                    g.setColor(new Color(b, b, b));
                    g.fillRect(c, r, 1, 1);
                }
            }
        }
        ImageIO.write(image, "png", path);
    }
}
