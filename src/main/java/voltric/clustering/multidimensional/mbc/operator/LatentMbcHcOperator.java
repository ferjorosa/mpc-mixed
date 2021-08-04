package voltric.clustering.multidimensional.mbc.operator;

import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.structure.latent.StructuralEM;
import voltric.model.DiscreteBayesNet;

public interface LatentMbcHcOperator {

    /**
     * Applies the operator to the Bayesian network and returns a new network. The application of the operator may result
     * in several possible models (i.e., increaseCardinality with several latent variables in the model). In that case,
     * the operator will return the highest scoring model.
     *
     * @param seedNet the initial latent MBC.
     * @param data the DataSet being used to learn the latent model once the operator is applied.
     * @param sem the Structural EM instance used to learn the model
     * @return the new BN. If the learning score equals {@code Double.MIN_VALUE} it measn the model hasn't been modified.
     */
    LearningResult<DiscreteBayesNet> apply(DiscreteBayesNet seedNet, DiscreteData data, StructuralEM sem);
}
