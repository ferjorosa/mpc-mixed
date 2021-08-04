package voltric.clustering.multidimensional.olcm.operator;

import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.model.DiscreteBayesNet;
import voltric.variables.DiscreteVariable;

/**
 * Created by equipo on 16/04/2018.
 */
public class DecreaseOlcmCard implements OlcmHcOperator{

    private int minCardinality;

    /**
     * Main constructor.
     *
     * @param minCardinality The minimum cardinality value
     */
    public DecreaseOlcmCard( int minCardinality){

        if(minCardinality < 2)
            throw  new IllegalArgumentException("The minimum cardinality value for a categorical variable is 2");

        this.minCardinality = minCardinality;
    }

    @Override
    public LearningResult<DiscreteBayesNet> apply(DiscreteBayesNet seedNet, DiscreteData data, AbstractEM em) {
        // The BN is copied to avoid modifying current object.
        DiscreteBayesNet clonedNet = seedNet.clone();

        double bestModelScore = -Double.MAX_VALUE; // Log-likelihood related scores are negative
        LearningResult<DiscreteBayesNet> bestModelResult = null;

        // Iteration through all the allowed BN nodes
        for(DiscreteVariable latentVar : clonedNet.getLatentVariables()) {

            // The cardinality of the LV must be lesser than the established maximum
            if(latentVar.getCardinality() > this.minCardinality) {

                clonedNet = clonedNet.decreaseCardinality(latentVar, 1);

                // After the LV has decreased its cardinality, the resulting model is learned. If its score is improved, the LV is stored
                LearningResult<DiscreteBayesNet> decreasedCardinalityResult = em.learnModel(clonedNet, data);

                if (decreasedCardinalityResult.getScoreValue() > bestModelScore) {
                    bestModelScore = decreasedCardinalityResult.getScoreValue();
                    bestModelResult = decreasedCardinalityResult;
                }

                // The cardinality is reversed for the next iteration to have the initial BN.
                // Given that we have previously increased the LV's cardinality, it is a new LV, so we have to access it by its name.
                /*
                    Note: There is no need to apply the EM algorithm or restore the initial parameters because it will be
                    called in the next iteration.
                */
                clonedNet = clonedNet.increaseCardinality(clonedNet.getLatentVariable(latentVar.getName()), 1);
            }
        }

        // If the model has been modified
        if(bestModelResult != null)
            return bestModelResult;

        return new LearningResult<>(null, bestModelScore, em.getScoreType());
    }
}
