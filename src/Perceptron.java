import java.util.*;

public class Perceptron {
    private final int dimension;
    private final double[] weights;
    private double threshold;
    private final double alpha;
    private final double beta;

    public Perceptron(int dimension, double alpha, double beta) {
        this.dimension = dimension;
        this.alpha = alpha;
        this.beta = beta;
        this.threshold = 0;
        this.weights = new double[dimension];

        Random rand = new Random();
        for (int i = 0; i < dimension; i++) {
            weights[i] = rand.nextDouble() * 0.5 - 0.25;
        }
    }

    public int predict(double[] inputs) {
        return computeNet(inputs) >= 0 ? 1 : 0;
    }

    public double computeNet(double[] inputs) {
        double net = 0;
        for (int i = 0; i < inputs.length; i++) {
            net += inputs[i] * weights[i];
        }
        net -= threshold;
        return net;
    }

    public void train(double[][] inputVec, int[] desiredOutputVec, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            int[] predictions = new int[inputVec.length];
            int errors = 0;

            for (int i = 0; i < inputVec.length; i++) {
                double[] inputs = inputVec[i];
                int expected = desiredOutputVec[i];
                int predicted = predict(inputs);
                predictions[i] = predicted;

                int error = expected - predicted;
                if (error != 0) {
                    errors++;
                    for (int j = 0; j < weights.length; j++) {
                        weights[j] += alpha * error * inputs[j];
                    }
                    threshold -= alpha * error;
                }
            }
//            double accuracy = EvaluationMetrics.measureAccuracy(toList(desiredOutputVec), toList(predictions));
//            System.out.println("Epoch " + (epoch + 1) + ": Errors = " + errors + ", Accuracy = " + (int) (accuracy * 100) + "%");
            if (errors == 0) break;
        }
    }

//    private List<Integer> toList(int[] arr) {
//        List<Integer> list = new ArrayList<>();
//        for (int val : arr) {
//            list.add(val);
//        }
//        return list;
//    }
}
