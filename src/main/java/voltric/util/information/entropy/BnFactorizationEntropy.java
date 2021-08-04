package voltric.util.information.entropy;

import voltric.inference.CliqueTreePropagation;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.potential.Function;
import voltric.util.Utils;
import voltric.variables.DiscreteVariable;

/**
 * Calculo de la entropia a partir de una distribucion conjunta factorizada utilizando una red bayesiana.
 *
 */
public class BnFactorizationEntropy {

    public static double compute(DiscreteBayesNet bayesNet){

        /**
         * Thanks to the BN factorization we can transform the calculation of the joint entropy into a sum of entropies of
         * its factors:
         *
         * H(X,Y,Z) = H(X|Pa(X)) + H(Y|Pa(Y)) + H(Z|Pa(Z))
         *
         * Given that P(X,Y,Z) = P(X|Pa(X)) * P(Y|Pa(Y)) * P(Z|Pa(Z))
         */

        double entropies = 0;

        CliqueTreePropagation cliqueTreePropagation = new CliqueTreePropagation(bayesNet);
        cliqueTreePropagation.propagate();

        for(DiscreteBeliefNode node: bayesNet.getNodes()) {
            Function cpt = node.getCpt();

            if(node.getParents().size() > 0){
                Function parentJoint = cliqueTreePropagation.computeBelief(node.getDiscreteParentVariables());
                entropies += compute(node.getVariable(), cpt, parentJoint);
            } else
                entropies += JointDistributionEntropy.compute(cpt);
        }

        return entropies;
    }

    public static double computeSumOfIndividualEntropies(DiscreteBayesNet bayesNet) {
        double sumOfIndividualEntropies = 0;

        CliqueTreePropagation cliqueTreePropagation = new CliqueTreePropagation(bayesNet);
        cliqueTreePropagation.propagate();

        for(DiscreteVariable variable: bayesNet.getVariables()) {
            Function marginal = cliqueTreePropagation.computeBelief(variable);
            sumOfIndividualEntropies += JointDistributionEntropy.compute(marginal);
        }

        return sumOfIndividualEntropies;
    }

    // TODO: Ahora que sabemos como obtener las probabilidades es conveniente probar si es mas rapida la formula de forma directa (ver wikipedia)
    // Por ahora utilizamos este metodo
    public static double computeConditional(DiscreteBayesNet xyBn, DiscreteBayesNet yBn){

        double Hxy = BnFactorizationEntropy.compute(xyBn); // H(X,Y)
        double Hy = BnFactorizationEntropy.compute(yBn); // H(Y)

        return Hxy - Hy;
    }

    // H(x|y) = \sum_{y \in Y} P(y) * \sum_{x \in X} P(x|y) log(P(x|y))
    public static double compute(DiscreteVariable variable, Function cpt, Function parentJoint) {

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
