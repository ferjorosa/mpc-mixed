package voltric.analysis;

import voltric.analysis.util.BnEntropy;
import voltric.model.DiscreteBayesNet;
import voltric.variables.DiscreteVariable;

import java.util.List;
import java.util.stream.Collectors;

/**
 * TODO: No tengo claro de si deberia dejar esta clase, pero quizas es interesante para reducir repeticion de codigo
 */
public class ConditionalEntropyPerCluster {

    /**
     * Version normalizada entre el valor de la entropia conjunta. El cual siempre deberia ser superior al de la entropia
     * condicionada.
     *
     * En teoria nos interesa que las entropias condicionadas sean lo mas bajas posibles, ya que eso significaria que el
     * cluster explica los datos adecuadamente.
     */
    public static double[] computeNormalized(DiscreteBayesNet bn, DiscreteVariable conditioningVar) {

        /* 1 - Calculamos las entropias condicionadas correspondientes a las hijas de la variable condicionante */
        double[] conditionalEntropies = BnEntropy.computeConditionalEntropies(bn, conditioningVar);

        /* 2 - Normalizamos al dividir entre la entropia del conjunto de variables formado por la variable condicionante y sus hijas */

        /* 2.1 - Seleccionamos aquellas que tengan como padre a la variable condicionante*/
        List<DiscreteVariable> subsetOfVars = bn.getNodes().stream()
                .filter(x-> x.hasParent(bn.getNode(conditioningVar)))
                .map(x->x.getVariable())
                .collect(Collectors.toList());
        /* 2.2 - Añadimos a la lista a la variable condicionante */
        subsetOfVars.add(conditioningVar);

        double subsetEntropy = BnEntropy.computeSubsetEntropy(bn, subsetOfVars);

        /* 2.3 - Dividimos las entropias condicionadas entre la entropia conjunta */
        for(int i=0;i<conditionalEntropies.length;i++)
            conditionalEntropies[i] = conditionalEntropies[i] / subsetEntropy;

        return conditionalEntropies;
    }
}
