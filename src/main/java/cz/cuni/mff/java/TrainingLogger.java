package cz.cuni.mff.java;

import java.io.PrintStream;

/**
 * Utility class for logging training progress and results
 * to the console during model training.
 */
public class TrainingLogger {
    private static final PrintStream console = System.out;

    public static void logStep(int epoch, int epochs, int stepProgress, double loss) {
        console.printf("\rEpoch: %d/%d, Progress: %d %%, Current Loss: %.6f", epoch, epochs, stepProgress, loss);
        console.flush(); // force output update in-place
    }

    public static void logEpoch(int epoch, int epochs, double loss, double accuracy, double f1) {
        console.println(); // new line after final step log
        console.printf("Epoch: %d/%d - Loss: %.6f, Accuracy: %.2f%% F1: %.4f %n%n", epoch, epochs, loss, accuracy * 100, f1);
    }

    public static void logTrainingComplete() {
        console.println("Training complete.");
    }
}
