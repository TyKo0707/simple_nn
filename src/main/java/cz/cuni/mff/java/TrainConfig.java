package cz.cuni.mff.java;

/**
 * Configurator for setting main parameters for training, like: architecture of hidden layers, epochs of training, ...
 */
public class TrainConfig {
    public static final int[] HIDDEN_LAYERS = {64, 32};
    public static final int OUTPUT_SIZE = 2;
    public static final double LEARNING_RATE = 0.005;
    public static final int EPOCHS = 10;
    public static final int BATCH_SIZE = 64;
}
