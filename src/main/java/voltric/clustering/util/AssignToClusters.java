package voltric.clustering.util;

import voltric.data.DiscreteData;
import voltric.data.DiscreteDataInstance;
import voltric.inference.CliqueTreePropagation;
import voltric.model.DiscreteBayesNet;
import voltric.potential.Function;
import voltric.util.Tuple;
import voltric.variables.DiscreteVariable;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by equipo on 18/12/2017.
 */
public class AssignToClusters {

    /**
     * Several LVs
     *
     * @param dataSet
     * @param ltm
     */

    public static List<Tuple<DiscreteDataInstance, List<double[]>>> assignDataCaseToClusters(DiscreteData dataSet, DiscreteBayesNet ltm) {

        List<Tuple<DiscreteDataInstance, List<double[]>>> partitionCpts = new ArrayList<>();
        Map<DiscreteVariable, Integer> evidence = new HashMap<>();

        DiscreteData projectedData = dataSet.project(ltm.getManifestVariables());

        // Creates a CliqueTreePropagation instance to do the inference
        CliqueTreePropagation inferenceEngine = new CliqueTreePropagation(ltm);
        for (DiscreteDataInstance dataCase : projectedData.getInstances()) {
            List<double[]> latentVarCells = new ArrayList<>();

            // Set the dataCase as evidence
            //TODO: We use the ltm variables because dataSet.var & ltm.var indexes may not coincide if the ltm was loaded or posteriorly created
            for (int i = 0; i < dataCase.getVariables().size(); i++)
                evidence.put(ltm.getManifestVariables().get(i), dataCase.getNumericValue(i));
            inferenceEngine.setEvidence(evidence);

            // Propagate the evidence
            inferenceEngine.propagate();
            // Request the values of the latentVariables
            for (DiscreteVariable latentVar : ltm.getLatentVariables())
                latentVarCells.add(inferenceEngine.computeBelief(latentVar).getCells());
            // Add the cpts associated with the dataCase to the array that will be returned
            partitionCpts.add(new Tuple<>(dataCase,latentVarCells));
        }
        return partitionCpts;
    }

    // Solo 1 LV
    public static List<Tuple<DiscreteDataInstance, double[]>> assignDataCaseToCluster(DiscreteData data, DiscreteBayesNet lcm) {

        if(lcm.getLatentVariables().size() > 1)
            throw new IllegalArgumentException("Solo 1 LV");

        DiscreteData projectedData = data.project(lcm.getManifestVariables());

        DiscreteVariable latentVar = lcm.getLatentVariables().get(0);
        List<Tuple<DiscreteDataInstance, double[]>> clusterAssignments = new ArrayList<>();
        Map<DiscreteVariable, Integer> evidence = new HashMap<>();

        // Creates a CliqueTreePropagation instance to do the inference
        CliqueTreePropagation inferenceEngine = new CliqueTreePropagation(lcm);
        for (DiscreteDataInstance dataCase : projectedData.getInstances()) {

            // Set the dataCase as evidence
            //TODO: We use the ltm variables because dataSet.var & ltm.var indexes may not coincide if the ltm was loaded or posteriorly created
            for (int i = 0; i < dataCase.getVariables().size(); i++)
                evidence.put(lcm.getManifestVariables().get(i), dataCase.getNumericValue(i));
            inferenceEngine.setEvidence(evidence);

            // Propagate the evidence
            inferenceEngine.propagate();
            Function latentVarFunc = inferenceEngine.computeBelief(latentVar);
            clusterAssignments.add(new Tuple<>(dataCase, latentVarFunc.getCells()));
        }

        return clusterAssignments;
    }
}