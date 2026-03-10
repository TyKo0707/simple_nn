package cz.cuni.mff.java;

import java.io.*;
import java.util.*;

/**
 * Utility class for handling dataset loading, splitting, and exporting
 * training metrics and ROC data from a (almost always) provided {@link SimpleNN} model.
 */
public class DataHandler {
    /**
     * Loads CSV-formatted data from the given file path.
     * The method assumes the first row is a header, skips it, and:
     * <ul>
     *   <li>Ignores the first column (e.g., an index or ID)</li>
     *   <li>Treats the last column as the target class label</li>
     * </ul>
     *
     * @param path path to the CSV file
     * @return a {@link DataSet} containing input vectors and labels
     * @throws RuntimeException if the file cannot be read
     */
    public static DataSet loadData(String path) {
        List<double[]> inputList = new ArrayList<>();
        List<Integer> targetList = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            boolean headerSkipped = false;
            while ((line = br.readLine()) != null) {
                if (!headerSkipped) {
                    headerSkipped = true;
                    continue;
                }
                String[] values = line.split(",");
                double[] input = new double[values.length - 2];
                int target = Integer.parseInt(values[values.length - 1]);

                for (int i = 1; i < values.length - 1; i++) {
                    input[i - 1] = Double.parseDouble(values[i]);
                }

                inputList.add(input);
                targetList.add(target);
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to load data: " + e.getMessage(), e);
        }

        double[][] inputs = inputList.toArray(new double[0][]);
        int[] targets = targetList.stream().mapToInt(Integer::intValue).toArray();
        return new DataSet(inputs, targets, inputs[0].length);
    }

    /**
     * Randomly splits the dataset into training and testing subsets using a fixed seed.
     *
     * @param data       the full dataset
     * @param trainRatio ratio of data to use for training (0–1)
     * @param seed       random seed to ensure reproducibility
     * @return a {@link SplitData} object with train/test splits
     */
    public static SplitData trainTestSplit(DataSet data, double trainRatio, long seed) {
        int total = data.inputs.length;
        int trainSize = (int) (total * trainRatio);

        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < total; i++) indices.add(i);
        Collections.shuffle(indices, new Random(seed));

        double[][] trainInputs = new double[trainSize][];
        int[] trainTargets = new int[trainSize];
        double[][] testInputs = new double[total - trainSize][];
        int[] testTargets = new int[total - trainSize];

        for (int i = 0; i < trainSize; i++) {
            trainInputs[i] = data.inputs[indices.get(i)];
            trainTargets[i] = data.targets[indices.get(i)];
        }

        for (int i = trainSize; i < total; i++) {
            int idx = indices.get(i);
            testInputs[i - trainSize] = data.inputs[idx];
            testTargets[i - trainSize] = data.targets[idx];
        }

        return new SplitData(trainInputs, trainTargets, testInputs, testTargets);
    }

    /**
     * Exports training metrics from a model to a CSV file.
     * Each row corresponds to an epoch with values for loss, accuracy, precision, and recall.
     *
     * @param model    the {@link SimpleNN} model to extract history from
     * @param filename path to save the metrics CSV
     * @throws RuntimeException if writing the file fails
     */
    public static void saveMetrics(SimpleNN model, String filename) {
        try (FileWriter writer = new FileWriter(filename)) {
            writer.write("Epoch,Loss,Accuracy,Precision,Recall\n");
            List<Double> loss = model.getLossHistory();
            List<Double> acc = model.getAccuracyHistory();
            List<Double> prec = model.getPrecisionHistory();
            List<Double> rec = model.getRecallHistory();
            for (int i = 0; i < loss.size(); i++) {
                writer.write(String.format("%d,%.6f,%.6f,%.6f,%.6f\n",
                        i + 1, loss.get(i), acc.get(i), prec.get(i), rec.get(i)));
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to save metrics: " + e.getMessage(), e);
        }
    }

    /**
     * Saves true labels and predicted scores from a model for ROC curve plotting.
     * Outputs two columns: true label and predicted probability for class 2.
     *
     * @param model    the {@link SimpleNN} model to extract ROC data from
     * @param filename path to save the ROC CSV
     * @throws RuntimeException if writing the file fails
     */
    public static void saveRocData(SimpleNN model, String filename) {
        try (FileWriter writer = new FileWriter(filename)) {
            writer.write("TrueLabel,PredictedScore\n");
            List<Integer> trueLabels = model.getTrueLabelHistory();
            List<Double> predictedScores = model.getPredictedScoreHistory();
            for (int i = 0; i < trueLabels.size(); i++) {
                writer.write(String.format("%d,%.6f\n", trueLabels.get(i), predictedScores.get(i)));
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to save ROC data: " + e.getMessage(), e);
        }
    }

    /**
     * Container for raw input features and corresponding class labels.
     */
    public static class DataSet {
        public final double[][] inputs;
        public final int[] targets;
        public final int inputSize;

        public DataSet(double[][] inputs, int[] targets, int inputSize) {
            this.inputs = inputs;
            this.targets = targets;
            this.inputSize = inputSize;
        }
    }

    /**
     * Container for split training and test datasets.
     */
    public static class SplitData {
        public final double[][] trainInputs;
        public final int[] trainTargets;
        public final double[][] testInputs;
        public final int[] testTargets;

        public SplitData(double[][] trainInputs, int[] trainTargets,
                         double[][] testInputs, int[] testTargets) {
            this.trainInputs = trainInputs;
            this.trainTargets = trainTargets;
            this.testInputs = testInputs;
            this.testTargets = testTargets;
        }
    }
}
