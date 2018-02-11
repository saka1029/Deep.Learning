package deep.learning.C2;

import static deep.learning.C2.C2_3_パーセプトロンの実装.*;
import static org.junit.Assert.*;

import org.junit.Test;

public class C2_5_多層パーセプトロン {

    public static int XOR(int x1, int x2) {
        int s1 = NAND(x1, x2);
        int s2 = OR(x1, x2);
        int y = AND2(s1, s2);
        return y;
    }

    @Test
    public void C2_5_2_XORゲートの実装() {
        assertEquals(0, XOR(0, 0));
        assertEquals(1, XOR(1, 0));
        assertEquals(1, XOR(0, 1));
        assertEquals(0, XOR(1, 1));
    }

}
