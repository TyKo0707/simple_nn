# SimpleNN Trainer

A lightweight Java-based neural network trainer for classification tasks.  
Includes CSV data loading, model training, evaluation, metrics export, and plotting.


---

## 🧠 Project Goal

Classify collision events from the LHC (CERN) into signal (Higgs decay) or background events using machine learning techniques in Java.  
The project is inspired by the [Higgs Boson Machine Learning Challenge](https://www.kaggle.com/competitions/higgs-boson/overview).

---

## 📦 Features

- Feedforward neural network with customizable architecture
- CSV data loading with train/test split
- Accuracy, precision, recall, and loss tracking
- ROC score export for visualization
- Plotting script integration
- Model serialization (save/load)

---

## 📁 Important Notes

> `root` refers to your project root directory (`path/to/HomeProject/`)

- To modify training settings → `TrainConfig.java`
- You can run tests using `mvn test` at root

---

## 🛠️ Setup & Build

You need to build a project, to do this, at root, run the following:

```bash
# Compile all Java source files
javac -d target $(find src/main/java -name "*.java")
```

You can then run examples using:
```bash
# Default: train, evaluate, save, and plot
java -cp target cz.cuni.mff.java.Main

# Train without evaluation
java -cp target cz.cuni.mff.java.Main --evaluate=false

# Train without plotting
java -cp target cz.cuni.mff.java.Main --plot=false

# Train without saving the model
java -cp target cz.cuni.mff.java.Main --save=false
```

## 📚 Javadoc
To regenerate Javadoc manually:
```bash
mvn javadoc:javadoc
```
Then open (from root):
```bash
open target/reports/apidocs/index.html      # macOS
xdg-open target/reports/apidocs/index.html  # Linux
start target/reports/apidocs/index.html     # Windows
```

## Example output
If you ran main function, you could expect the following results:
```bash
...
Epoch: 10/10, Progress: 100 %, Current Loss: 0.046332
Epoch: 10/10 - Loss: 0.067298, Accuracy: 97.57% F1: 0.9655 

Evaluation:
=== Evaluation Results ===
Loss      : 0.0637
Accuracy  : 97.90%
Precision : 95.11%
Recall    : 98.94%
==========================
```

### Metrics Explained

The model is evaluated using common classification metrics:

- **Loss**: Difference between predicted outputs and real target values (used for training in backpropagation algorithm)
- **Accuracy**: Proportion of correctly predicted labels.
- **Precision**: Percentage of predicted positives that are actual positives.
- **Recall**: Percentage of actual positives correctly identified.
- **F1-score**: Harmonic mean of precision and recall, useful when classes are imbalanced.

And the plot is stored at: `src/data/all_metrics.png`
