package cz.cuni.mff.java;

import java.util.Arrays;

/**
 * Provides common activation functions and their derivatives
 * for use in neural network layers.
 */
public class ActivationFunctions {

    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double sigmoidDerivative(double x) {
        double sigmoid = sigmoid(x);
        return sigmoid * (1 - sigmoid);
    }

    public static double relu(double x) {
        return Math.max(0, x);
    }

    public static double reluDerivative(double x) {
        return x > 0 ? 1 : 0;
    }

    public static double tanh(double x) {
        return Math.tanh(x);
    }

    public static double tanhDerivative(double x) {
        double tanh = tanh(x);
        return 1 - tanh * tanh;
    }

    public static double[] relu(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = relu(x[i]);
        }
        return result;
    }

    public static double[] softmax(double[] x) {
        double max = Arrays.stream(x).max().getAsDouble();
        double sum = 0;
        double[] expVals = new double[x.length];

        for (int i = 0; i < x.length; i++) {
            expVals[i] = Math.exp(x[i] - max);
            sum += expVals[i];
        }

        for (int i = 0; i < x.length; i++) {
            expVals[i] /= sum;
        }

        return expVals;
    }
}
