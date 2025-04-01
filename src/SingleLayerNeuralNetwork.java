import java.util.*;

public class SingleLayerNeuralNetwork {
    private final List<Perceptron> neurons;
    private final double alpha;
    private final double beta;

    public SingleLayerNeuralNetwork(int inputDimension, int numClasses, double alpha, double beta) {
        this.alpha = alpha;
        this.beta = beta;
        this.neurons = new ArrayList<>();

        for (int i = 0; i < numClasses; i++) {
            neurons.add(new Perceptron(inputDimension, alpha, beta));
        }
    }

    public void trainLayer(double[][] inputs, int[] labels, int epochs) {
        int numClasses = neurons.size();

        for (int i = 0; i < numClasses; i++) {
            int[] binaryLabels = new int[labels.length];
            for (int j = 0; j < labels.length; j++) {
                //for each class' index, we convert original multi-class labels into a binary label vector.
                //Each perceptron act as a binary classifier responsible for recognizing its own class.
                binaryLabels[j] = (labels[j] == i) ? 1 : 0;
            }
            neurons.get(i).train(inputs, binaryLabels, epochs);
        }
    }

    public int predict(double[] input) {
        List<Integer> activatedClasses = new ArrayList<>();
        List<Double> netValues = new ArrayList<>();

        for (int i = 0; i < neurons.size(); i++) {
            //Each Perceptron's input
            int activated = neurons.get(i).predict(input);
            if (activated == 1) {
                activatedClasses.add(i);
                //collect net values of all perceptrons that returned 1
                netValues.add(neurons.get(i).computeNet(input));
            }
        }

        if (activatedClasses.isEmpty()) {
            return -1;
        }
        else if (activatedClasses.size() == 1) {
            return activatedClasses.getFirst();
        }
        else {
            //for starting the lowest possible. We will compare the net values
            double maxNet = Double.NEGATIVE_INFINITY;
            int selectedClass = -1;
            for (int i = 0; i < activatedClasses.size(); i++) {
                if (netValues.get(i) > maxNet) {
                    maxNet = netValues.get(i);
                    selectedClass = activatedClasses.get(i);
                }
            }
            return selectedClass;
        }
    }
}
