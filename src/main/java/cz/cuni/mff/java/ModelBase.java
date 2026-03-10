package cz.cuni.mff.java;

import java.io.IOException;
import java.util.Random;
import java.util.Arrays;

/**
 * Abstract base class for neural network models.
 * <p>
 * Defines the common interface for training, inference,
 * serialization, and utility functions like shuffling and one-hot encoding.
 */
public abstract class ModelBase {
    protected final Random rand = new Random();

    public abstract double[][] forward(double[][] inputs);
    public abstract void train(double[][] trainData, int[] trainTargets,
                               double[][] testData, int[] testTargets,
                               int epochs, int batchSize) throws IOException;
    protected abstract void initializeWeights();

    public abstract void saveModel(String filePath) throws IOException;
    public abstract void loadModel(String filePath) throws IOException;

    protected int[] shuffleIndices(int size) {
        int[] indices = new int[size];
        for (int i = 0; i < size; i++) indices[i] = i;
        for (int i = size - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        return indices;
    }

    protected double[] oneHot(int label, int classes) {
        double[] oneHot = new double[classes];
        oneHot[label - 1] = 1.0;
        return oneHot;
    }
}
