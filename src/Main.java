import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) throws IOException {
        LanguageDataLoader.load("resources/train_data.csv");

        List<double[]> trainVectors = new ArrayList<>();
        for (String text : LanguageDataLoader.texts) {
            trainVectors.add(TextVectorizer.vectorize(text));
        }

        double[][] trainInputs = trainVectors.toArray(new double[0][]);
        int[] trainLabels = LanguageDataLoader.labels.stream().mapToInt(i -> i).toArray();

        SingleLayerNeuralNetwork net = new SingleLayerNeuralNetwork(26, 3, 0.1, 0.1);
        net.trainLayer(trainInputs, trainLabels, 50);

        LanguageDataLoader.texts.clear();
        LanguageDataLoader.labels.clear();
        LanguageDataLoader.load("resources/test_data.csv");

        List<double[]> testVectors = new ArrayList<>();
        for (String text : LanguageDataLoader.texts) {
            testVectors.add(TextVectorizer.vectorize(text));
        }

        double[][] testInputs = testVectors.toArray(new double[0][]);
        int[] testLabels = LanguageDataLoader.labels.stream().mapToInt(i -> i).toArray();

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
