package deep.learning.common;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.Closeable;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

/**
 * 簡易的なグラフ作成用のクラスです。
 * 結果はPNGファイルとして保存します。
 */
public class GraphImage implements Closeable {

    public final int width, height;
    public final double minX, minY, maxX, maxY;
    public final double dx, dy;
    public final Graphics2D graphics;
    final BufferedImage image;

    static final int DEFAULT_DOT_SIZE = 7;

    /**
     * グラフを新規に作成します。
     * 座標値はdoubleで指定し、X座標は右方向、Y座標は上方向になります。
     *
     * @param width 作成する画像の幅をピクセル数で指定します。
     * @param height 作成する画像の幅をピクセル数で指定します。
     * @param minX 描画する最小の座標値（左下）のX座標を指定します。
     * @param minY 描画する最小の座標値（左下）のY座標を指定します。
     * @param maxX 描画する最大の座標値（右上）のX座標を指定します。
     * @param maxY 描画する最大の座標値（右上）のY座標を指定します。
     */
    public GraphImage(int width, int height, double minX, double minY, double maxX, double maxY) {
        this.width = width;
        this.height = height;
        this.minX = minX;
        this.minY = minY;
        this.maxX = maxX;
        this.maxY = maxY;
        this.dx = width / (maxX - minX);
        this.dy = height / (maxY - minY);
        this.image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        this.graphics = this.image.createGraphics();
        this.graphics.setColor(new Color(240, 240, 240));
        this.graphics.fillRect(0, 0, width, height);
        this.graphics.setColor(Color.BLACK);
        // X座標を描画します。
        line(minX, 0, maxX, 0);
        // Y座標を描画します。
        line(0, minY, 0, maxY);
        this.graphics.setColor(Color.RED);
    }

    int x(double x) {
        return (int)((x - minX) * dx);
    }

    int y(double y) {
        return height - (int)((y - minY) * dy);
    }

    /**
     * 描画する色を設定します。
     */
    public void color(Color c) {
        graphics.setColor(c);
    }

    /**
     * テキストを描画します。
     */
    public void text(String str, double x, double y) {
        graphics.drawString(str, x(x), y(y));
    }

    /**
     * 点をプロットします。
     * @size 点の大きさをピクセル数で指定します。
     */
    public void plot(double x, double y, int size) {
        int half = size / 2;
        graphics.fillOval(x(x) - half, y(y) - half, size, size);
    }

    /**
     * デフォルトの大きさの点をプロットします。
     */
    public void plot(double x, double y) {
        plot(x, y, DEFAULT_DOT_SIZE);
    }

    /**
     * 線を描画します。
     * @param x1 始点のX座標を指定します。
     * @param y1 始点のY座標を指定します。
     * @param x2 終点のX座標を指定します。
     * @param y2 終点のY座標を指定します。
     */
    public void line(double x1, double y1, double x2, double y2) {
        graphics.drawLine(x(x1), y(y1), x(x2), y(y2));
    }

    /**
     * 指定したファイルにPNG画像として出力します。
     * @param output
     * @throws IOException
     */
    public void writeTo(File output) throws IOException {
        ImageIO.write(image, "png", output);
    }

    @Override
    public void close() {
        graphics.dispose();
    }
}
