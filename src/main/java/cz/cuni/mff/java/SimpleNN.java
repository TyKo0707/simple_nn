package cz.cuni.mff.java;

import cz.cuni.mff.utils.NNUtils;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.io.Serializable;

/**
 * A basic feedforward neural network that supports backpropagation, batch training, several hidden layers,
 * and recording evaluation metrics.
 * <p>
 * Model serialisation, accuracy/precision/recall reporting,
 * and the ability to visualise ROC score history are all supported by this class.
 */
public class SimpleNN extends ModelBase implements Serializable {
    private static final long serialVersionUID = 1L;

    // Model architecture
    private int inputSize;
    private int outputSize;
    private double learningRate;
    private int[] layerSizes;
    private final List<double[][]> weights = new ArrayList<>();
    private final List<double[]> biases = new ArrayList<>();

    // Training history
    private final List<Double> lossHistory = new ArrayList<>();
    private final List<Double> accuracyHistory = new ArrayList<>();
    private final List<Double> precisionHistory = new ArrayList<>();
    private final List<Double> recallHistory = new ArrayList<>();

    // ROC/AUC export data
    private final List<Integer> trueLabelHistory = new ArrayList<>();
    private final List<Double> predictedScoreHistory = new ArrayList<>();

    public SimpleNN(int inputSize, int[] hiddenLayers, int outputSize, double learningRate) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.learningRate = learningRate;

        this.layerSizes = new int[hiddenLayers.length + 2];
        this.layerSizes[0] = inputSize;
        System.arraycopy(hiddenLayers, 0, this.layerSizes, 1, hiddenLayers.length);
        this.layerSizes[layerSizes.length - 1] = outputSize;

        initializeWeights();
    }

    @Override
    protected void initializeWeights() {
        for (int i = 0; i < layerSizes.length - 1; i++) {
            int inSize = layerSizes[i];
            int outSize = layerSizes[i + 1];

            double[][] w = new double[inSize][outSize];
            double[] b = new double[outSize];

            for (int j = 0; j < inSize; j++) {
                for (int k = 0; k < outSize; k++) {
                    w[j][k] = rand.nextDouble() * 0.2 - 0.1;
                }
            }
            Arrays.fill(b, 0);
            weights.add(w);
            biases.add(b);
        }
    }

    /**
     * Performs a forward pass on a batch of input vectors.
     *
     * @param inputs 2D array of input samples (shape: batchSize × inputSize)
     * @return network output for each input (softmax scores)
     */
    @Override
    public double[][] forward(double[][] inputs) {
        double[][] output = new double[inputs.length][];
        for (int i = 0; i < inputs.length; i++) {
            output[i] = forwardSingle(inputs[i]);
        }
        return output;
    }

    private double[] forwardSingle(double[] input) {
        double[] activation = input;
        for (int i = 0; i < weights.size(); i++) {
            double[] z = NNUtils.matrixVectorMultiply(weights.get(i), activation, biases.get(i));
            activation = (i == weights.size() - 1)
                    ? ActivationFunctions.softmax(z)
                    : ActivationFunctions.relu(z);
        }
        return activation;
    }

    /**
     * Trains the network using mini-batch stochastic gradient descent.
     *
     * @param trainData    input samples for training
     * @param trainTargets corresponding class labels
     * @param testData     input samples for validation/testing
     * @param testTargets  corresponding class labels for validation
     * @param epochs       number of training epochs
     * @param batchSize    size of each mini-batch
     * @throws IOException if metric logging fails
     */
    @Override
    public void train(double[][] trainData, int[] trainTargets,
                      double[][] testData, int[] testTargets,
                      int epochs, int batchSize) throws IOException {
        int numSamples = trainData.length;

        for (int epoch = 0; epoch < epochs; epoch++) {
            int[] permutation = shuffleIndices(numSamples);
            double totalLoss = 0;
            int steps = 0;

            for (int i = 0; i < numSamples; i += batchSize) {
                int actualBatchSize = Math.min(batchSize, numSamples - i);
                int[] batchIndices = Arrays.copyOfRange(permutation, i, i + actualBatchSize);
                double[][] batchData = new double[actualBatchSize][inputSize];
                double[][] batchTargets = new double[actualBatchSize][outputSize];

                for (int j = 0; j < actualBatchSize; j++) {
                    batchData[j] = trainData[batchIndices[j]];
                    batchTargets[j] = oneHot(trainTargets[batchIndices[j]], outputSize);
                }

                double[][] predictions = forward(batchData);
                double batchLoss = NNUtils.crossEntropyLoss(predictions, batchTargets);
                totalLoss += batchLoss;
                steps++;

                backpropagate(batchData, batchTargets);
                TrainingLogger.logStep(epoch + 1, epochs, (int) (100 * i / numSamples) + 1, batchLoss);
            }

            EvaluationResult testResult = evaluate(testData, testTargets, epoch == epochs - 1);

            lossHistory.add(testResult.loss);
            accuracyHistory.add(testResult.accuracy);
            precisionHistory.add(testResult.precision);
            recallHistory.add(testResult.recall);

            TrainingLogger.logEpoch(epoch + 1, epochs,totalLoss / steps, testResult.accuracy,
                    2 * testResult.precision * testResult.recall / (testResult.precision + testResult.recall + 1e-8));
        }

        TrainingLogger.logTrainingComplete();
    }

    /**
     * Saves the current model state to a file using Java serialization.
     *
     * @param filePath destination file path
     * @throws IOException if writing fails
     */
    @Override
    public void saveModel(String filePath) throws IOException {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filePath))) {
            out.writeObject(this);
        }
    }

    /**
     * Loads the model state from a file and replaces all internal parameters.
     *
     * @param filePath source file path
     * @throws IOException if deserialization fails
     */
    @Override
    public void loadModel(String filePath) throws IOException {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(filePath))) {
            SimpleNN loaded = (SimpleNN) in.readObject();

            // Copy fields from loaded into this
            this.inputSize = loaded.inputSize;
            this.outputSize = loaded.outputSize;
            this.learningRate = loaded.learningRate;
            this.layerSizes = loaded.layerSizes;
            this.weights.clear();
            this.weights.addAll(loaded.weights);
            this.biases.clear();
            this.biases.addAll(loaded.biases);
            this.lossHistory.clear();
            this.lossHistory.addAll(loaded.lossHistory);
            this.accuracyHistory.clear();
            this.accuracyHistory.addAll(loaded.accuracyHistory);
            this.precisionHistory.clear();
            this.precisionHistory.addAll(loaded.precisionHistory);
            this.recallHistory.clear();
            this.recallHistory.addAll(loaded.recallHistory);
            this.trueLabelHistory.clear();
            this.trueLabelHistory.addAll(loaded.trueLabelHistory);
            this.predictedScoreHistory.clear();
            this.predictedScoreHistory.addAll(loaded.predictedScoreHistory);
        } catch (ClassNotFoundException e) {
            throw new IOException("Failed to deserialize model: " + e.getMessage(), e);
        }
    }

    private void backpropagate(double[][] batchData, double[][] batchTargets) {
        int batchSize = batchData.length;
        int numLayers = weights.size();

        List<double[][]> activations = new ArrayList<>();
        List<double[][]> zs = new ArrayList<>();
        List<double[][]> deltas = new ArrayList<>(numLayers);

        forwardPass(batchData, activations, zs, batchSize, numLayers);
        initializeDeltas(deltas, batchSize, numLayers);
        computeOutputDelta(deltas, activations, batchTargets, batchSize, numLayers);
        computeHiddenDeltas(deltas, zs, numLayers, batchSize);
        applyGradients(deltas, activations, batchSize, numLayers);
    }

    private void forwardPass(double[][] batchData, List<double[][]> activations, List<double[][]> zs, int batchSize, int numLayers) {
        activations.add(batchData);

        for (int l = 0; l < numLayers; l++) {
            double[][] currentZ = new double[batchSize][];
            double[][] currentA = new double[batchSize][];

            for (int i = 0; i < batchSize; i++) {
                currentZ[i] = NNUtils.matrixVectorMultiply(weights.get(l), activations.get(l)[i], biases.get(l));
                currentA[i] = (l == numLayers - 1)
                        ? ActivationFunctions.softmax(currentZ[i])
                        : ActivationFunctions.relu(currentZ[i]);
            }

            zs.add(currentZ);
            activations.add(currentA);
        }
    }

    private void initializeDeltas(List<double[][]> deltas, int batchSize, int numLayers) {
        for (int i = 0; i < numLayers; i++) {
            deltas.add(new double[batchSize][layerSizes[i + 1]]);
        }
    }

    private void computeOutputDelta(List<double[][]> deltas, List<double[][]> activations, double[][] batchTargets, int batchSize, int numLayers) {
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                deltas.get(numLayers - 1)[i][j] = batchTargets[i][j] - activations.get(numLayers)[i][j];
            }
        }
    }

    private void computeHiddenDeltas(List<double[][]> deltas, List<double[][]> zs, int numLayers, int batchSize) {
        for (int l = numLayers - 2; l >= 0; l--) {
            int currentSize = layerSizes[l + 1];
            int nextSize = layerSizes[l + 2];

            for (int i = 0; i < batchSize; i++) {
                for (int j = 0; j < currentSize; j++) {
                    double sum = 0;
                    for (int k = 0; k < nextSize; k++) {
                        sum += deltas.get(l + 1)[i][k] * weights.get(l + 1)[j][k];
                    }
                    double z = zs.get(l)[i][j];
                    deltas.get(l)[i][j] = sum * ActivationFunctions.reluDerivative(z);
                }
            }
        }
    }

    private void applyGradients(List<double[][]> deltas, List<double[][]> activations, int batchSize, int numLayers) {
        for (int l = 0; l < numLayers; l++) {
            double[][] gradW = weights.get(l);
            double[] gradB = biases.get(l);
            int inSize = gradW.length;
            int outSize = gradW[0].length;

            for (int i = 0; i < inSize; i++) {
                for (int j = 0; j < outSize; j++) {
                    double sum = 0;
                    for (int b = 0; b < batchSize; b++) {
                        sum += deltas.get(l)[b][j] * activations.get(l)[b][i];
                    }
                    gradW[i][j] += learningRate * sum / batchSize;
                }
            }

            for (int j = 0; j < outSize; j++) {
                double sum = 0;
                for (int b = 0; b < batchSize; b++) {
                    sum += deltas.get(l)[b][j];
                }
                gradB[j] += learningRate * sum / batchSize;
            }
        }
    }

    public List<Double> getLossHistory() {
        return lossHistory;
    }

    public List<Double> getAccuracyHistory() {
        return accuracyHistory;
    }

    public List<Double> getPrecisionHistory() {
        return precisionHistory;
    }

    public List<Double> getRecallHistory() {
        return recallHistory;
    }

    public List<Integer> getTrueLabelHistory() {
        return trueLabelHistory;
    }

    public List<Double> getPredictedScoreHistory() {
        return predictedScoreHistory;
    }

    /**
     * Evaluates the model on the given dataset and computes key classification metrics.
     *
     * @param data    input features
     * @param targets true class labels
     * @param saveRoc whether to record ROC scores for visualization
     * @return an {@link EvaluationResult} containing loss, accuracy, precision, and recall
     */
    public EvaluationResult evaluate(double[][] data, int[] targets, boolean saveRoc) {
        double[][] predictions = forward(data);
        int[] predictedLabels = new int[predictions.length];

        for (int i = 0; i < predictions.length; i++) {
            predictedLabels[i] = NNUtils.argMax(predictions[i]) + 1;
        }

        double acc = Metrics.accuracy(targets, predictedLabels);
        double prec = Metrics.precision(targets, predictedLabels);
        double rec = Metrics.recall(targets, predictedLabels);
        double loss = NNUtils.crossEntropyLoss(predictions,
                Arrays.stream(targets).mapToObj(t -> oneHot(t, outputSize)).toArray(double[][]::new));

        if (saveRoc) {
            trueLabelHistory.clear();
            predictedScoreHistory.clear();
            for (int i = 0; i < predictions.length; i++) {
                trueLabelHistory.add(targets[i]);
                predictedScoreHistory.add(predictions[i][1]); // prob for class 2
            }
        }

        return new EvaluationResult(loss, acc, prec, rec);
    }

    /**
     * A simple container for evaluation results including loss, accuracy,
     * precision, and recall.
     */
    public static class EvaluationResult {
        public final double loss, accuracy, precision, recall;

        public EvaluationResult(double loss, double accuracy, double precision, double recall) {
            this.loss = loss;
            this.accuracy = accuracy;
            this.precision = precision;
            this.recall = recall;
        }

        @Override
        public String toString() {
            return String.format(
                    "=== Evaluation Results ===\n" +
                            "Loss      : %.4f\n" +
                            "Accuracy  : %.2f%%\n" +
                            "Precision : %.2f%%\n" +
                            "Recall    : %.2f%%\n" +
                            "==========================",
                    loss, accuracy * 100, precision * 100, recall * 100
            );
        }
    }

    public void copyFrom(SimpleNN other) {
        this.inputSize = other.inputSize;
        this.outputSize = other.outputSize;
        this.learningRate = other.learningRate;
        this.layerSizes = other.layerSizes.clone();
        this.weights.clear();
        this.weights.addAll(other.weights);
        this.biases.clear();
        this.biases.addAll(other.biases);
        this.lossHistory.clear();
        this.lossHistory.addAll(other.lossHistory);
        this.accuracyHistory.clear();
        this.accuracyHistory.addAll(other.accuracyHistory);
        this.precisionHistory.clear();
        this.precisionHistory.addAll(other.precisionHistory);
        this.recallHistory.clear();
        this.recallHistory.addAll(other.recallHistory);
        this.trueLabelHistory.clear();
        this.trueLabelHistory.addAll(other.trueLabelHistory);
        this.predictedScoreHistory.clear();
        this.predictedScoreHistory.addAll(other.predictedScoreHistory);
    }

}
