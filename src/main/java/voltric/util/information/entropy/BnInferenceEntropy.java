package voltric.util.information.entropy;

import voltric.data.DiscreteData;
import voltric.data.DiscreteDataInstance;
import voltric.inference.CliqueTreePropagation;
import voltric.util.Utils;
import voltric.variables.DiscreteVariable;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by equipo on 14/11/2017.
 */
public class BnInferenceEntropy {

    public static double compute(CliqueTreePropagation ctp, DiscreteData data){

        if(!data.getVariables().containsAll(ctp.getBayesNet().getVariables()))
            throw new IllegalArgumentException("All the variables inside the CTP must belong to the DataSet");

        // TODO: Es necesario proyectar porque sino se pasan instancias repetidas a la evidencia al no haberlas proyectado
        DiscreteData projectedData = data.project(ctp.getBayesNet().getVariables());

        double entropy = 0;
        Map<DiscreteVariable, Integer> evidenceValues = new HashMap<>();
        for (DiscreteDataInstance dataCase : projectedData.getInstances()) {

            evidenceValues.clear();
            for(DiscreteVariable var: ctp.getBayesNet().getVariables())
                evidenceValues.put(var, dataCase.getNumericValue(var));

            ctp.setEvidence(evidenceValues);
            double instanceProbability = ctp.propagate();
            entropy += instanceProbability * Utils.log(instanceProbability);
        }

        return -entropy;
    }
}
