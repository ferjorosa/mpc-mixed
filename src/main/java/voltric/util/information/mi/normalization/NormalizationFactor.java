package voltric.util.information.mi.normalization;

import voltric.data.DiscreteData;
import voltric.inference.CliqueTreePropagation;
import voltric.model.DiscreteBayesNet;
import voltric.variables.DiscreteVariable;

import java.util.List;

/**
 * Created by equipo on 25/07/2017.
 */
public interface NormalizationFactor {

    double partialNormalizeMI(double mi, double px, double py, double pxy);

    // frequency-countered normalization (according to FrequencyCounteredMI)
    double normalizeMI(double mi, List<DiscreteVariable> x, List<DiscreteVariable> y, DiscreteData dataSet);

    double normalizeCMI(double cmi, List<DiscreteVariable> x, List<DiscreteVariable> y, List<DiscreteVariable> condVars, DiscreteData data);

    /**************************** BN Factorization *********************************/
    double normalizeMI(double mi, DiscreteBayesNet xBn, DiscreteBayesNet yBn, DiscreteBayesNet xyBn);

    /**************************** BN Inference *********************************/
    double normalizeMI(double mi, CliqueTreePropagation xCtp, CliqueTreePropagation yCtp, CliqueTreePropagation xyCtp, DiscreteData data);
}
