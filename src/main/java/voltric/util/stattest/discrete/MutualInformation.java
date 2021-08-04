package voltric.util.stattest.discrete;

import voltric.data.DiscreteData;
import voltric.potential.Function;
import voltric.util.information.mi.MI;
import voltric.variables.DiscreteVariable;

import java.util.List;
import java.util.Map;

/**
 * Created by fernando on 26/08/17.
 */
public class MutualInformation implements DiscreteStatisticalTest {

    /** {@inheritDoc} */
    @Override
    public double computePairwise(DiscreteVariable x, DiscreteVariable y, DiscreteData dataSet) {
        return MI.computePairwise(x, y, dataSet);
    }

    /** {@inheritDoc} */
    @Override
    public double computePairwiseParallel(DiscreteVariable x, DiscreteVariable y, DiscreteData dataSet) {
        return MI.computePairwiseParallel(x, y, dataSet);
    }

    /** {@inheritDoc} */
    @Override
    public double computePairwise(DiscreteVariable x, List<DiscreteVariable> y, DiscreteData dataSet) {
        return MI.computePairwise(x, y, dataSet);
    }

    /** {@inheritDoc} */
    @Override
    public double computePairwise(List<DiscreteVariable> x, List<DiscreteVariable> y, DiscreteData dataSet) {
        return MI.computePairwise(x, y , dataSet);
    }

    /** {@inheritDoc} */
    @Override
    public Map<DiscreteVariable, Map<DiscreteVariable, Double>> computePairwise(List<DiscreteVariable> variables, DiscreteData dataSet) {
        return MI.computePairwise(variables, dataSet);
    }

    /** {@inheritDoc} */
    @Override
    public Map<DiscreteVariable, Map<DiscreteVariable, Double>> computePairwiseParallel(List<DiscreteVariable> variables, DiscreteData dataSet) {
        return MI.computePairwiseParallel(variables, dataSet);
    }

    /** {@inheritDoc} */
    @Override
    public double computeConditional(DiscreteVariable x, DiscreteVariable y, DiscreteVariable condVar, DiscreteData data) {
        return MI.computeConditional(x, y, condVar, data);
    }

    /** {@inheritDoc} */
    @Override
    public double computeConditional(DiscreteVariable x, DiscreteVariable y, List<DiscreteVariable> condVars, DiscreteData data) {
        return MI.computeConditional(x, y, condVars, data);
    }

    /** {@inheritDoc} */
    @Override
    public double computeConditional(List<DiscreteVariable> x, List<DiscreteVariable> y, List<DiscreteVariable> condVars, DiscreteData data) {
        return MI.computeConditional(x, y, condVars, data);
    }

    @Override
    public double computeConditional(Function dist, DiscreteVariable condVar) {
        return MI.computeConditional(dist, condVar);
    }
}
