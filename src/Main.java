import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) throws IOException {
// 1. Load data
        LanguageDataLoader.load("resources/lang_data.csv");

// 2. Vectorize all texts
        List<double[]> inputVectors = new ArrayList<>();
        for (String text : LanguageDataLoader.texts) {
            inputVectors.add(TextVectorizer.vectorize(text));
        }
        double[][] allInputs = inputVectors.toArray(new double[0][]);
        int[] allLabels = LanguageDataLoader.labels.stream().mapToInt(i -> i).toArray();

// 3. Split data 70-30
        List<double[]> trainInputsList = new ArrayList<>();
        List<Integer> trainLabelsList = new ArrayList<>();
        List<double[]> testInputsList = new ArrayList<>();
        List<Integer> testLabelsList = new ArrayList<>();

        LanguageDataLoader.splitDataset(allInputs, allLabels, 0.4, trainInputsList, trainLabelsList, testInputsList, testLabelsList);

// Convert to arrays
        double[][] trainInputs = trainInputsList.toArray(new double[0][]);
        int[] trainLabels = trainLabelsList.stream().mapToInt(i -> i).toArray();
        double[][] testInputs = testInputsList.toArray(new double[0][]);
        int[] testLabels = testLabelsList.stream().mapToInt(i -> i).toArray();

// 4. Train
        SingleLayerNeuralNetwork net = new SingleLayerNeuralNetwork(26, 3, 0.1, 0.1);
        net.trainLayer(trainInputs, trainLabels, 50);

// 5. Predict and Evaluate
        int[] predictedLabels = new int[testInputs.length];
        for (int i = 0; i < testInputs.length; i++) {
            predictedLabels[i] = net.predict(testInputs[i]);
        }

        evaluateModel(testLabels, predictedLabels, 3);

    }

    public static void evaluateModel(int[] trueLabels, int[] predictedLabels, int numClasses) {
        double accuracy = EvaluationMetrics.measureAccuracy(toList(trueLabels), toList(predictedLabels));
        System.out.printf("Overall Accuracy: %.2f%%\n", accuracy * 100);

        for (int classId = 0; classId < numClasses; classId++) {
            List<Integer> binaryTrue = new ArrayList<>();
            List<Integer> binaryPred = new ArrayList<>();

            for (int i = 0; i < trueLabels.length; i++) {
                binaryTrue.add(trueLabels[i] == classId ? 1 : 0);
                binaryPred.add(predictedLabels[i] == classId ? 1 : 0);
            }

            double precision = EvaluationMetrics.precision(binaryTrue, binaryPred);
            double recall = EvaluationMetrics.recall(binaryTrue, binaryPred);
            double f1 = EvaluationMetrics.measureFMeasure(binaryTrue, binaryPred);

            System.out.printf("Class %d — Precision: %.2f, Recall: %.2f, F1-Score: %.2f\n",
                    classId, precision, recall, f1);
        }
    }

    private static List<Integer> toList(int[] array) {
        List<Integer> list = new ArrayList<>();
        for (int val : array) {
            list.add(val);
        }
        return list;
    }
}
