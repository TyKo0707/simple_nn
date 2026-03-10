package cz.cuni.mff.utils;

/**
 * Utility class for common neural network operations such as
 * matrix-vector multiplication, loss computation, and argmax.
 */
public class NNUtils {

    public static double[] matrixVectorMultiply(double[][] matrix, double[] vector, double[] bias) {
        double[] result = new double[matrix[0].length];
        for (int j = 0; j < matrix[0].length; j++) {
            result[j] = bias[j];
            for (int i = 0; i < matrix.length; i++) {
                result[j] += matrix[i][j] * vector[i];
            }
        }
        return result;
    }

    public static double crossEntropyLoss(double[][] predictions, double[][] targets) {
        double loss = 0;
        for (int i = 0; i < predictions.length; i++) {
            for (int j = 0; j < predictions[i].length; j++) {
                loss -= targets[i][j] * Math.log(predictions[i][j] + 1e-9);
            }
        }
        return loss / predictions.length;
    }

    public static int argMax(double[] array) {
        int maxIdx = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIdx]) {
                maxIdx = i;
            }
        }
        return maxIdx;
    }
}
