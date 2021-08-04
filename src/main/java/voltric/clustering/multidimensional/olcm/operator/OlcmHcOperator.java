package voltric.clustering.multidimensional.olcm.operator;

import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.model.DiscreteBayesNet;

/**
 * Created by equipo on 16/04/2018.
 */
public interface OlcmHcOperator {

    /**
     * Applies the operator to the Bayesian network and returns a new network. In  case there are various possible
     * result models, the best one will be returned (i.e Arc Operators).
     *
     * @param seedNet the initial BN.
     * @param data the DataSet being used to learn the OLCM after the operator is applied.
     * @param em the em algorithm used to learn the OLCM
     * @return the new BN. If the learning score equals {@code Double.MIN_VALUE} it measn the model hasn't been modified.
     */
    LearningResult<DiscreteBayesNet> apply(DiscreteBayesNet seedNet, DiscreteData data, AbstractEM em);
}
