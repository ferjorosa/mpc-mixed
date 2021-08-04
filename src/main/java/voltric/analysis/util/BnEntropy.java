package voltric.analysis.util;

import voltric.inference.CliqueTreePropagation;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.potential.Function;
import voltric.util.Utils;
import voltric.util.information.entropy.JointDistributionEntropy;
import voltric.variables.DiscreteVariable;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Esta clase contiene como calcular la entropia condicionada de una BN para VALORES ESPECIFICOS.
 *
 * Por ahora esta pensada para modelos unidimensionales de clustering ya que se va a considerar unicamente los hijos
 * de la variable condicionante. Tengo que ver como afectaria este approach para modelos multidimensionales. El procedimiento es
 * el mismo que para obtener la entropia conjunta de una parte de la red, simplemente que hemos dado evidencia
 * a una variable padre.
 *
 */
public class BnEntropy {

    /**
     *  COMO FUNCTIONA: Se le pasa una BN y una variable perteneciente a dicha BN, despues se calcula la entropia
     *  condicionada de las variablaes hijas para cada uno de los valores de la misma.
     *
     *  Formula de ejemplo para una red cuando condicionamos C = 1.
     *
     *  H(X_{1}, H(X_{2}),..., H(X_{n}) \mid C = 1) = \sum_{i=1}^{n} H(X_{i} \mid Pa{X_{i}}, C = 1)
     *
     * @param bn
     * @param conditioningVar
     * @return
     */
    // TODO: Ha sido diseñado principalmente para LCMs. Para modelos mas complejos se podria considerar el Markov Blanket
    // TODO: de la variable condicionante o todas las variables que tengan un camino no dirigido hasta ella, ya que
    // TODO: les afecta indirectamente.
    // COUNTER al statement previo: Tambien es verdad que solo consideramos a sus hijas ya que son las que en teoria pertenecen
    // al clustering de dicha variable, variables ajenas afectan al clustering pero a lo mejor no nos interesa si las separa bien
    public static double[] computeConditionalEntropies(DiscreteBayesNet bn, DiscreteVariable conditioningVar) {

        if(!bn.containsVar(conditioningVar))
            throw new IllegalArgumentException("The conditioning var must belong to the BN");

        DiscreteBeliefNode conditioningNode = bn.getNode(conditioningVar);
        double[] entropies = new double[conditioningVar.getStates().size()];

        /* 1 - Seleccionamos los nodos hijos de la variable condicionante*/
        List<DiscreteBeliefNode> childrenNodes = bn.getNodes().stream()
                .filter(x->x.hasParent(conditioningNode))
                .collect(Collectors.toList());

        /* 2 - Calculamos la entropia condicionada a cada uno de los estados de la variable condicionante */

        /* 2.1 - Creamos el objeto de inferencia */
        CliqueTreePropagation cliqueTreePropagation = new CliqueTreePropagation(bn);
        Map<DiscreteVariable, Integer> evidence = new HashMap<>();

        /* 2.2 - Iteramos por los estados y oara cada uno de ellos calculas la entropia condicionada */
        for(int i = 0; i < conditioningVar.getStates().size(); i++){

            /* 2.2.1 - Ponemos el estado como evidencia para la inferencia*/
            evidence.clear();
            evidence.put(conditioningVar, i);
            cliqueTreePropagation.setEvidence(evidence);

            double conditionalEntropy = 0;

            /* 2.2.2 - Iteramos por cada uno de los hijos y hacemos dos cosas para calcular la entropia condicionada:
            *
            *       1) Proyectamos la CPT del hijo con el estado de la variable condicionante
            *
            *       2) Mediante inferencia estimamos la distribucion conjunta de los padres menos la var condicionante (ya que forma parte de la evidencia)
            * */
            for(DiscreteBeliefNode childNode: childrenNodes){
                Function projectedCpt = childNode.getCpt().project(conditioningVar, i);

                // En caso de que tenga mas de un padre (considerando que la conditioningVar es uno de ellos)
                if(childNode.getParents().size() > 1){
                    List<DiscreteVariable> parentVarsMinusConditioningVar = childNode.getDiscreteParentVariables().stream()
                            .filter(x->!x.getName().equals(conditioningVar.getName()))
                            .collect(Collectors.toList());

                    Function parentJoint = cliqueTreePropagation.computeBelief(parentVarsMinusConditioningVar);
                    conditionalEntropy += compute(childNode.getVariable(), projectedCpt, parentJoint);

                // En caso de que solo tenga como padre a la variable condicionante
                } else
                    conditionalEntropy += JointDistributionEntropy.compute(projectedCpt);
            }

            entropies[i] = conditionalEntropy;
        }

        return entropies;
    }

    /**
     * Este metodo es equivalente al de {@link voltric.util.information.entropy.BnFactorizationEntropy}, simplemente
     * seleccionamos un subset de variables sobre el que iterar en vez de todas las de la red.
     */
    public static double computeSubsetEntropy(DiscreteBayesNet bn, List<DiscreteVariable> subsetOfVariables) {

        if(!bn.containsVars(subsetOfVariables))
            throw new IllegalArgumentException("All the subset vars must belong to the BN");

        double entropies = 0;

        CliqueTreePropagation cliqueTreePropagation = new CliqueTreePropagation(bn);
        cliqueTreePropagation.propagate();

        List<DiscreteBeliefNode> nodes = bn.getNodes().stream()
                .filter(x->subsetOfVariables.contains(x.getVariable()))
                .collect(Collectors.toList());

        for(DiscreteBeliefNode node: nodes) {
            Function cpt = node.getCpt();

            if(node.getParents().size() > 0){
                Function parentJoint = cliqueTreePropagation.computeBelief(node.getDiscreteParentVariables());
                entropies += compute(node.getVariable(), cpt, parentJoint);
            } else
                entropies += JointDistributionEntropy.compute(cpt);
        }

        return entropies;
    }

    /**
     * Esta funcion ya se encuentra en BnFactorizationEntropy, la añadimos aqui por facilidad, el procedimiento es
     * el mismo que para obtener la entropia conjunta de una parte de la red, simplemente que hemos dado evidencia
     * a una variable padre
     */
    // H(x|y) = \sum_{y \in Y} P(y) * \sum_{x \in X} P(x|y) log(P(x|y))
    private static double compute(DiscreteVariable variable, Function cpt, Function parentJoint) {

        int[] cptIndexMap = new int[cpt.getDimension()];
        int[] cptStates = new int[cpt.getDimension()];
        int[] parentStates = new int[parentJoint.getDimension()];

        for(int i=0; i < cpt.getDimension(); i++){
            if(i == 0)
                // En nuestro map la variable de la CPT siempre es la primera
                cptIndexMap[i] = cpt.getVariables().indexOf(variable);
            else
                // El indice en la CPT de la variable parent de la parent joint
                cptIndexMap[i] = cpt.getVariables().indexOf(parentJoint.getVariables().get(i-1));
        }

        double condEntropy = 0.0;
        double condProb;

        for(int i = 0; i < parentJoint.getDomainSize(); i++){
            double jointvalue = parentJoint.getCells()[i];
            double condInstEntropy = 0.0;
            parentJoint.computeStates(i, parentStates);

            for(int j = 0; j < variable.getCardinality(); j++){

                // Calcula el array de estados de las variables de la CPT en el orden de la misma
                for(int k=0; k < cptStates.length; k++){
                    if(k==0)
                        cptStates[cptIndexMap[k]] = j;
                    else
                        // Como hemos establecido que la hija siempre va a ir delante, k siempre sera 1 valor mayor para parentStates
                        cptStates[cptIndexMap[k]] = parentStates[k-1];
                }
                // Obtiene el indice dentro del array que representa la CPT para el conjunto de estados
                int cptIndex = cpt.computeIndex(cptStates);
                // Obtiene la entropia condicionada a partir de la probabilidad condicionada
                condProb =  cpt.getCells()[cptIndex];
                condInstEntropy += condProb * Utils.log(condProb);// Utils.log, because log(0) == 0 for this case
            }
            condEntropy += jointvalue * condInstEntropy;
        }

        return -condEntropy;
    }
}
