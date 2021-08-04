package voltric.util.stattest.discrete;

import voltric.data.DiscreteData;
import voltric.potential.Function;
import voltric.variables.DiscreteVariable;

import java.util.List;
import java.util.Map;

/**
 * Created by fernando on 26/08/17.
 */
public interface DiscreteStatisticalTest {

    double computePairwise(DiscreteVariable x, DiscreteVariable y, DiscreteData dataSet);

    double computePairwiseParallel(DiscreteVariable x, DiscreteVariable y, DiscreteData dataSet);

    double computePairwise(DiscreteVariable x, List<DiscreteVariable> y, DiscreteData dataSet);

    double computePairwise(List<DiscreteVariable> x, List<DiscreteVariable> y, DiscreteData dataSet);

    Map<DiscreteVariable, Map<DiscreteVariable, Double>> computePairwise(List<DiscreteVariable> variables, DiscreteData dataSet);

    Map<DiscreteVariable, Map<DiscreteVariable, Double>> computePairwiseParallel(List<DiscreteVariable> variables, DiscreteData dataSet);

    double computeConditional(DiscreteVariable x, DiscreteVariable y, DiscreteVariable condVar, DiscreteData data);

    double computeConditional(DiscreteVariable x, DiscreteVariable y, List<DiscreteVariable> condVars, DiscreteData data);

    double computeConditional(List<DiscreteVariable> x, List<DiscreteVariable> y, List<DiscreteVariable> condVars, DiscreteData data);

    double computeConditional(Function dist, DiscreteVariable condVar);
}
