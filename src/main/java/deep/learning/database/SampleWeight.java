package deep.learning.database;

import java.io.*;
import java.util.*;

import org.nd4j.linalg.api.ndarray.*;
import org.nd4j.linalg.factory.*;

public class SampleWeight {

    public static Map<String, INDArray> read(File input) throws IOException {
        Map<String, INDArray> weights = new HashMap<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(input))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] header = line.split("\\s+");
                INDArray value;
                if (header.length == 2) {
                    int rows = Integer.parseInt(header[1]);
                    value = Nd4j.create(rows);
                    weights.put(header[0], value);
                    for (int r = 0; r < rows; ++r)
                        value.putScalar(r, Double.parseDouble(reader.readLine()));
                } else if (header.length == 3) {
                    int rows = Integer.parseInt(header[1]);
                    int cols = Integer.parseInt(header[2]);
                    value = Nd4j.create(rows, cols);
                    weights.put(header[0], value);
                    for (int r = 0; r < rows; ++r)
                        for (int c = 0; c < cols; ++c)
                            value.putScalar(r, c, Double.parseDouble(reader.readLine()));
                } else
                    throw new IOException("Invalid format: " + line);
            }
        }
        return weights;
    }

}
