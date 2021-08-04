package voltric.clustering.multidimensional.mbc.operator;

import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.structure.latent.StructuralEM;
import voltric.model.DiscreteBayesNet;
import voltric.variables.DiscreteVariable;

// TODO: Cambiar maxCardinality por minCardinality. El codigo esta pensado para incrementar la cardinalidad, mirar condicion del while
public class DecreaseLatentCardinality implements LatentMbcHcOperator {

    /**
     * The maximum allowed cardinality value.
     */
    private int maxCardinality;

    /**
     * Main constructor.
     *
     * @param maxCardinality The maximum allowed cardinality value.
     */
    public DecreaseLatentCardinality(int maxCardinality) {
        this.maxCardinality = maxCardinality;
    }

    @Override
    public LearningResult<DiscreteBayesNet> apply(DiscreteBayesNet seedNet, DiscreteData data, StructuralEM sem) {

        /*
            No clonamos la red al inicio como en los operadores OLHC ya que cada cambio en la variable latente tiene
            consecuencias en la estructura que serian costosos de revertir al final de cada iteracion, de esta manera
            es mucho mas simple.
         */
        double bestModelScore = -Double.MAX_VALUE; // Log-likelihood related scores are negative
        LearningResult<DiscreteBayesNet> bestModelResult = null;

        /* Iteration through all the allowed BN nodes */
        for(DiscreteVariable latentVar : seedNet.getLatentVariables()) {

            // 1 - The BN is copied to avoid modifying current object.
            DiscreteBayesNet clonedNet = seedNet.clone();

            // 2 - The cardinality of the LV must be lesser than the established maximum
            if (latentVar.getCardinality() < this.maxCardinality) {

                clonedNet = clonedNet.decreaseCardinality(clonedNet.getLatentVariable(latentVar.getName()), 1);

                // NOTA: Dado que hemos decrementado la cardinalidad de la variable, hemos creado otro objeto con diferente
                // equals y hashcode, ya que posee diferente vector de estados. Es por ello necesario actualizar las
                // restricciones del SEM
                DiscreteVariable newLatentVar = clonedNet.getLatentVariable(latentVar.getName());
                sem.addLatentVarWithRestrictions(newLatentVar, latentVar);

                /*
                    Once the cardinality of the LV has been increased, the resulting model is learned with the Structural EM
                    to see if changes in the arcs must be applied. But first, it is necessary to generate a initial set
                    of parameters for the SEM with the EM algorithm (the first run)
                 */
                LearningResult<DiscreteBayesNet> initialModelForSEM = sem.getEm().learnModel(clonedNet, data);

                LearningResult<DiscreteBayesNet> semResult = sem.learnModel(initialModelForSEM, data);

                // 3 - If the resulting model learned with SEM is better than the current one, we store the model
                if (semResult.getScoreValue() > bestModelScore) {
                    bestModelScore = semResult.getScoreValue();
                    bestModelResult = semResult;
                }

                // 4 - Una vez se ha comprobado si el modelo mejora o no el score actual, se pasa a la siguiente iteracion
                // Aunque no se revierten cambios en la estructura por utilizar una copia en cada iteracion, es
                // necesario eliminar las restricciones del SEM para la nueva LV que habiamos añadido
                sem.removeLatentVar(newLatentVar);
            }
        }

        /* Finalmente si el modelo ha sido modificado, lo devolvemos */
        if(bestModelResult != null)
            return bestModelResult;

        /* En caso contrario, devolvemos un modelo "falso" con score infinitamente malo */
        return new LearningResult<>(null, bestModelScore, sem.getScoreType());
    }
}
