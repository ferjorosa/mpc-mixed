package voltric.util.information.totalcorrelation;

import voltric.data.DiscreteData;
import voltric.util.information.entropy.FrequencyCountedEntropy;

/**
 * Developed by Watanabe (1960)
 */
public class TotalCorrelation {

    /**
     * Returns the entropy of the specified discrete data set.
     *
     * @param data the discrete dataset.
     * @return the entropy value.
     */
    public static double compute(DiscreteData data){
        double jointEntropy = FrequencyCountedEntropy.compute(data);
        double sumIndividualEntropies = FrequencyCountedEntropy.computeSumOfIndividualEntropies(data);
        return jointEntropy - sumIndividualEntropies;
    }
}
