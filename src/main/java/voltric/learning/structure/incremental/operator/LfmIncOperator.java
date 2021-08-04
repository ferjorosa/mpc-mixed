package voltric.learning.structure.incremental.operator;

import eu.amidst.extension.util.tuple.Tuple3;
import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.model.DiscreteBayesNet;
import voltric.variables.DiscreteVariable;

import java.util.List;
import java.util.PriorityQueue;

public interface LfmIncOperator {

    /**
     * Apply the operator to each of currentSet pairs of variables and return the highest scoring model.
     */
    Tuple3<DiscreteVariable, DiscreteVariable, LearningResult<DiscreteBayesNet>> apply(List<String> currentset,
                                                                                       DiscreteBayesNet bayesNet,
                                                                                       DiscreteData data);

    /**
     * Apply the operator to each of the selected pairs of variables and return the highest scoring model.
     */
    Tuple3<DiscreteVariable, DiscreteVariable, LearningResult<DiscreteBayesNet>> apply(PriorityQueue<Tuple3<String, String, Double>> selectedTriples,
                                                                                       DiscreteBayesNet bayesNet,
                                                                                       DiscreteData data);
}
