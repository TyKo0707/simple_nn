package cz.cuni.mff.java;

import cz.cuni.mff.java.DataHandler.*;
import java.io.*;
import java.util.*;

/**
 * The Main class is the main point for training/evaluating/plotting results of a neural network model.
 * <p>
 * Command-line arguments:
 * <ul>
 *   <li><b>--evaluate</b> (default: true) - Whether to evaluate the model on test set after training.</li>
 *   <li><b>--plot</b> (default: true) - Whether to run the Python script to plot metrics and ROC curve.</li>
 *   <li><b>--save</b> (default: true) - Whether to save the trained model locally.</li>
 *   <li><b>--modelPath</b> (default: {@code src/models/model.ser}) - Path to load or save the model.</li>
 *   <li><b>--dataPath</b> (default: {@code src/data/processed_training.csv}) - Path to the dataset.</li>
 *   <li><b>--splitRatio</b> (default: 0.8) - Ratio of training data to use (remaining is for testing).</li>
 * </ul>
 *
 * <p>
 * After training or loading the model, it optionally evaluates the model and generates visual metrics.
 *
 * <p>Example of usage:
 * <pre>{@code
 * java -cp target cz.cuni.mff.java.Main --evaluate=false --modelPath=out/model.ser
 * }</pre>
 */

public class Main {
    /**
     * Program entry point. Parses command-line arguments, loads or trains a model,
     * optionally evaluates it, saves metrics, and visualizes results.
     *
     * @param args command-line arguments (e.g., {@code --evaluate=false})
     * @throws IOException if reading/writing model or dataset fails
     */
    public static void main(String[] args) throws IOException {
        Map<String, String> flags = parseArgs(args);
        boolean doEvaluate = !flags.getOrDefault("evaluate", "true").equalsIgnoreCase("false");
        boolean doPlot = !flags.getOrDefault("plot", "true").equalsIgnoreCase("false");
        boolean saveModel = flags.getOrDefault("save", "true").equalsIgnoreCase("true");
        String modelPath = flags.getOrDefault("modelPath", "src/models/model.ser");
        String dataPath = flags.getOrDefault("dataPath", "src/data/processed_training.csv");
        double splitRatio = Double.parseDouble(flags.getOrDefault("splitRatio", String.valueOf(0.8)));

        DataSet fullData = DataHandler.loadData(dataPath);
        SplitData split = DataHandler.trainTestSplit(fullData, splitRatio, 42);

        SimpleNN model = new SimpleNN(
                fullData.inputSize,
                TrainConfig.HIDDEN_LAYERS,
                TrainConfig.OUTPUT_SIZE,
                TrainConfig.LEARNING_RATE
        );

        loadOrTrainModel(model, split, modelPath, saveModel);

        if (doEvaluate) {
            System.out.println("Evaluation:");
            Object result = model.evaluate(split.testInputs, split.testTargets, true);
            System.out.println(result + "\n");
        }

        DataHandler.saveMetrics(model, "src/data/metrics.csv");
        DataHandler.saveRocData(model, "src/data/roc_data.csv");

        if (doPlot) {
            System.out.println("Plotting the metrics.");
            runPlottingScript("src/data/metrics.csv", "src/data/roc_data.csv");
        } else {
            System.out.println("Metrics and ROC saved. Use --plot to visualize.");
        }
    }

    /**
     * Loads a trained model if some exists, otherwise trains a new model and saves it (if flag is set).
     *
     * @param model     the {@link SimpleNN} instance to load into or train
     * @param data      the dataset split into train/test parts
     * @param modelPath the path to load or save the serialized model
     * @param saveModel if true, saves the trained model after training
     * @throws IOException if deserialization or saving fails
     */
    public static void loadOrTrainModel(SimpleNN model, SplitData data, String modelPath, boolean saveModel) throws IOException {
        File modelFile = new File(modelPath);
        if (modelFile.exists()) {
            System.out.println("Model found, loading from file..." + "\n");
            try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(modelFile))) {
                SimpleNN loadedModel = (SimpleNN) ois.readObject();
                model.copyFrom(loadedModel);
            } catch (ClassNotFoundException e) {
                throw new IOException("Model class not found during deserialization", e);
            }
        } else {
            System.out.println("No model found, training new model...");
            model.train(
                    data.trainInputs,
                    data.trainTargets,
                    data.testInputs,
                    data.testTargets,
                    TrainConfig.EPOCHS,
                    TrainConfig.BATCH_SIZE
            );

            if (saveModel) {
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelPath))) {
                    oos.writeObject(model);
                }
                System.out.println("Model saved to " + modelPath + "\n");
            }
        }
    }

    /**
     * Runs a Python script to plot training metrics and ROC curves.
     *
     * @param metricsFile the path to the metrics CSV file
     * @param rocFile     the path to the ROC curve CSV file
     * @throws RuntimeException if the Python script fails to run
     */
    public static void runPlottingScript(String metricsFile, String rocFile) {
        try {
            ProcessBuilder pb = new ProcessBuilder("python3", "src/main/python/plot_metrics.py", metricsFile, rocFile);
            pb.inheritIO();
            Process process = pb.start();
            process.waitFor();
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException("Failed to run Python plotting script: " + e.getMessage(), e);
        }
    }

    public static Map<String, String> parseArgs(String[] args) {
        Map<String, String> map = new HashMap<>();
        for (String arg : args) {
            if (arg.startsWith("--")) {
                String[] parts = arg.substring(2).split("=", 2);
                map.put(parts[0], parts.length > 1 ? parts[1] : "true");
            }
        }
        return map;
    }
}
