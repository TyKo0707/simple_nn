package cz.cuni.mff.java;

/**
 * Provides basic evaluation metrics for binary classification.
 */
public class Metrics {
    public static double accuracy(int[] trueLabels, int[] predictedLabels) {
        int correct = 0;
        for (int i = 0; i < trueLabels.length; i++) {
            if (trueLabels[i] == predictedLabels[i]) {
                correct++;
            }
        }
        return (double) correct / trueLabels.length;
    }

    public static double precision(int[] trueLabels, int[] predictedLabels) {
        int tp = 0, fp = 0;
        for (int i = 0; i < trueLabels.length; i++) {
            if (predictedLabels[i] == 1) {
                if (trueLabels[i] == 1) {
                    tp++;
                } else {
                    fp++;
                }
            }
        }
        return tp + fp == 0 ? 0 : (double) tp / (tp + fp);
    }

    public static double recall(int[] trueLabels, int[] predictedLabels) {
        int tp = 0, fn = 0;
        for (int i = 0; i < trueLabels.length; i++) {
            if (trueLabels[i] == 1) {
                if (predictedLabels[i] == 1) {
                    tp++;
                } else {
                    fn++;
                }
            }
        }
        return tp + fn == 0 ? 0 : (double) tp / (tp + fn);
    }
}
