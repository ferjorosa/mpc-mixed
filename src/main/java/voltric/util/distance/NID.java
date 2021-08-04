package voltric.util.distance;

import voltric.model.DiscreteBayesNet;
import voltric.util.information.mi.NMI;
import voltric.util.information.mi.normalization.NMImax;

/**
 * Created by equipo on 09/04/2018.
 */
public class NID {

    public static double calculate(DiscreteBayesNet firstBn, DiscreteBayesNet secondBn, DiscreteBayesNet combinedBn) {
        return 1 - NMI.computePairwise(firstBn, secondBn, combinedBn, new NMImax());
    }
}
