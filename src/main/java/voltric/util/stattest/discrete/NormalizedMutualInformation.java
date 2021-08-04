package voltric.util.stattest.discrete;

import org.apache.commons.lang3.NotImplementedException;
import voltric.data.DiscreteData;
import voltric.potential.Function;
import voltric.util.information.mi.NMI;
import voltric.util.information.mi.normalization.NormalizationFactor;
import voltric.variables.DiscreteVariable;

import java.util.List;
import java.util.Map;

;

/**
 * Created by fernando on 26/08/17.
 */
public class NormalizedMutualInformation implements DiscreteStatisticalTest {

    private NormalizationFactor normalizationFactor;

    public NormalizedMutualInformation(NormalizationFactor normalizationFactor){
        this.normalizationFactor = normalizationFactor;
    }

    /** {@inheritDoc} */
    @Override
    public double computePairwise(DiscreteVariable x, DiscreteVariable y, DiscreteData dataSet) {
        return NMI.computePairwise(x, y, dataSet, this.normalizationFactor);
    }

    /** {@inheritDoc} */
    @Override
    public double computePairwiseParallel(DiscreteVariable x, DiscreteVariable y, DiscreteData dataSet) {
        return NMI.computePairwiseParallel(x, y, dataSet, this.normalizationFactor);
    }

    /** {@inheritDoc} */
    @Override
    public double computePairwise(DiscreteVariable x, List<DiscreteVariable> y, DiscreteData dataSet) {
        return NMI.computePairwise(x, y, dataSet, this.normalizationFactor);
    }

    /** {@inheritDoc} */
    @Override
    public double computePairwise(List<DiscreteVariable> x, List<DiscreteVariable> y, DiscreteData dataSet) {
        return NMI.computePairwise(x, y, dataSet, this.normalizationFactor);
    }

    /** {@inheritDoc} */
    @Override
    public Map<DiscreteVariable, Map<DiscreteVariable, Double>> computePairwise(List<DiscreteVariable> variables, DiscreteData dataSet) {
        return NMI.computePairwise(variables, dataSet, this.normalizationFactor);
    }

    /** {@inheritDoc} */
    @Override
    public Map<DiscreteVariable, Map<DiscreteVariable, Double>> computePairwiseParallel(List<DiscreteVariable> variables, DiscreteData dataSet) {
        return NMI.computePairwiseParallel(variables, dataSet, this.normalizationFactor);
    }

    @Override
    public double computeConditional(DiscreteVariable x, DiscreteVariable y, DiscreteVariable condVar, DiscreteData data) {
        return NMI.computeConditional(x, y, condVar, data, this.normalizationFactor);
    }

    @Override
    public double computeConditional(DiscreteVariable x, DiscreteVariable y, List<DiscreteVariable> condVars, DiscreteData data) {
        return NMI.computeConditional(x, y, condVars, data, this.normalizationFactor);
    }

    @Override
    public double computeConditional(List<DiscreteVariable> x, List<DiscreteVariable> y, List<DiscreteVariable> condVars, DiscreteData data) {
        return NMI.computeConditional(x, y, condVars, data, this.normalizationFactor);
    }

    @Override
    public double computeConditional(Function dist, DiscreteVariable condVar) {
        throw new NotImplementedException("Nope");
    }
}
