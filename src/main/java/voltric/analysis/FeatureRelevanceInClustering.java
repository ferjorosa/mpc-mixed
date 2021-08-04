package voltric.analysis;

import voltric.inference.CliqueTreePropagation;
import voltric.model.DiscreteBayesNet;
import voltric.potential.Function;
import voltric.util.information.mi.MI;
import voltric.variables.DiscreteVariable;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * TODO: El objetivo es obtener la relevancia de cada una de las variables con respecto a la variable de clustering.
 * La manera de calculararla se hace mediante JointMI.java
 * Si no existe un arco entre la varable y la variable de clustering, le asignamos el valor 0
 */
public class FeatureRelevanceInClustering {

    public static Map<DiscreteVariable, Double> calculateMI(DiscreteBayesNet bn, DiscreteVariable clusteringVar) {

        if(!bn.getLatentVariables().contains(clusteringVar))
            throw new IllegalArgumentException("The clustering variable must belong to the model");

        Map<DiscreteVariable, Double> misWithClusterVar = new HashMap<>();

        CliqueTreePropagation ctp = new CliqueTreePropagation(bn);
        ctp.propagate();

        /* Iteramos por cada una de las variables manifest del modelo */
        for(DiscreteVariable manifestVar: bn.getManifestVariables()) {

            // Si NO tiene como padre a la variable de clustering le asignamos el valor cero
            // TODO: para el caso multidimensional habria que considerar que no tuviera NINGUN padre relacionado con la var de clustering
            if (!bn.getNode(manifestVar).hasParent(bn.getNode(clusteringVar))) {
                misWithClusterVar.put(manifestVar, 0.0);

            // Por el contrario...
            } else {

                /* Computamos la distribucion conjunta entre la variable hija y la variable de clustering */
                List<DiscreteVariable> jointDistVars = new ArrayList<>();
                jointDistVars.add(manifestVar);
                jointDistVars.add(clusteringVar);

                Function joint = ctp.computeBelief(jointDistVars);

                /* Computamos la MI entre ambas variables */
                double mi = MI.computePairwise(joint);

                misWithClusterVar.put(manifestVar, mi);
            }
        }

        return misWithClusterVar;
    }
}
