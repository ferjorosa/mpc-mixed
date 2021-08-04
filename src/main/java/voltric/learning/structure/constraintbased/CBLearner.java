package voltric.learning.structure.constraintbased;

import voltric.data.DiscreteData;
import voltric.graph.UndirectedGraph;
import voltric.variables.DiscreteVariable;

/**
 * Constraint-based learner that returns an undirected graph
 */
public interface CBLearner {

    int getMaxParentOrder();

    UndirectedGraph<DiscreteVariable> learnSkeleton(DiscreteData data);
}
