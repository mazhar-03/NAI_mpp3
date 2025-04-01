import java.util.Locale;

public class TextVectorizer {

    // Converts input text into a normalized 26D vector of letter probabilities
    public static double[] vectorize(String text) {

        double[] frequencies = new double[26];
        text = text.toLowerCase(Locale.ROOT).replaceAll("[^a-z]", "");

        int totalLetters = text.length();
        if (totalLetters == 0) {
            return frequencies; // all zero vector
        }

        for (char c : text.toCharArray()) {
            if (c >= 'a' && c <= 'z') {
                frequencies[c - 'a']++;
            }
        }

        //normalizing
        for (int i = 0; i < 26; i++) {
            frequencies[i] /= totalLetters;
        }

        return frequencies;
    }
}
