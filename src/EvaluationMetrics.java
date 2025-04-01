import java.util.List;

// TRUE POSITIVE: real -> 1 && predicted -> 1
// FALSE POSITIVE: real -> 0 && predicted -> 1
// TRUE NEGATIVE: real -> 0 && predicted -> 0
// FALSE NEGATIVE: real -> 1 && predicted -> 0

public class EvaluationMetrics {
    public static double measureAccuracy(List<Integer> realClasses, List<Integer> predictedClasses) {
        int correct = 0;
        for (int i = 0; i < realClasses.size(); i++) {
            if (realClasses.get(i).equals(predictedClasses.get(i))) {
                correct++;
            }
        }
        return (double) correct / realClasses.size();
    }

    public static double precision(List<Integer> realClasses, List<Integer> predictedClasses) {
        int truePositive = 0, falsePositive = 0;
        for (int i = 0; i < realClasses.size(); i++) {
            if (predictedClasses.get(i) == 1) {
                if (realClasses.get(i) == 1) {
                    truePositive++;
                } else {
                    falsePositive++;
                }
            }
        }
        return truePositive / (double) (truePositive + falsePositive);
    }

    public static double recall(List<Integer> realClasses, List<Integer> predictedClasses) {
        int truePositive = 0, falseNegative = 0;
        for (int i = 0; i < realClasses.size(); i++) {
            if (realClasses.get(i) == 1) {
                if (predictedClasses.get(i) == 1) {
                    truePositive++;
                } else {
                    falseNegative++;
                }
            }
        }
        return truePositive / (double) (truePositive + falseNegative);
    }

    public static double measureFMeasure(List<Integer> realClasses, List<Integer> predictedClasses) {
        double precision = precision(realClasses, predictedClasses);
        double recall = recall(realClasses, predictedClasses);
        return 2 * (precision * recall) / (precision + recall);
    }
}
