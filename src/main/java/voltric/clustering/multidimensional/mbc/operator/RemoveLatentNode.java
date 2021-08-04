package voltric.clustering.multidimensional.mbc.operator;

import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.structure.latent.StructuralEM;
import voltric.model.DiscreteBayesNet;
import voltric.variables.DiscreteVariable;

/**
 * Elimina un nodo latente y utiliza la nueva estructura como base para un proceso de Structural EM.
 *
 * Si bien la eliminacion de variables latentes suele ser un proceso algo complejo, en el caso de MBCs latentes generales
 * vamos a seguir el approach mas simple, que es simplemente eliminar el nodo y los arcos asociados para luego pasarselo
 * al SEM.
 *
 * En este caso a diferencia de AddLatentNode, no vamos a actualizar el SEM internamente al borrar un nodo, ya que no
 * tiene consecuencias negativas. Simplemente se hará en {@link voltric.clustering.multidimensional.mbc.LatentMbcHcWithSEM}
 * en caso de que esta operacion fuese escogida como la que mejor score proporciona.
 */
public class RemoveLatentNode implements LatentMbcHcOperator {

    @Override
    public LearningResult<DiscreteBayesNet> apply(DiscreteBayesNet seedNet, DiscreteData data, StructuralEM sem) {

        /*
            No clonamos la red al inicio como en los operadores OLHC ya que cada cambio en la variable latente tiene
            consecuencias en la estructura que serian costosos de revertir al final de cada iteracion, de esta manera
            es mucho mas simple.
         */
        double bestModelScore = -Double.MAX_VALUE;
        LearningResult<DiscreteBayesNet> bestModelResult = null;

        /*
            Iteramos por el conjunto de variables latentes en caso de que haya mas de 1 y los eliminamos 1 a 1 alternativamente
            para ver que tipo de estructura se genera con el Structural EM.
        */
        if(seedNet.getLatentVariables().size() > 1) {
            for (DiscreteVariable latentVariable : seedNet.getLatentVariables()) {

                // 1 - Copiamos la red inicial para no modificarla y poder volver a empezar de cero con cada iteracion
                DiscreteBayesNet clonedNet = seedNet.clone();

                // 2 - Eliminamos el nodo latente en la red clonada
                clonedNet.removeNode(clonedNet.getNode(latentVariable));

                // 3 - Aprendemos los parametros del modelo con EM
                LearningResult<DiscreteBayesNet> initialModelForSEM = sem.getEm().learnModel(clonedNet, data);

                // 4 - Tomamos la red previamente aprendida como modelo inicial para el Structural EM
                LearningResult<DiscreteBayesNet> semResult = sem.learnModel(initialModelForSEM, data);

                // 5 - Una vez aprendido el modelo con SEM, comprobamos que mejore el score actual y si es asi, lo almacenamos
                if (semResult.getScoreValue() > bestModelScore) {
                    bestModelScore = semResult.getScoreValue();
                    bestModelResult = semResult;
                }

                // 6 - Una vez se ha comprobado si el modelo mejora o no el score actual, se pasa a la siguiente iteracion
                // No se revierten cambios ya que se utilizara una copia fresca de seedNet
            }
        }

        /* Finalmente si el modelo ha sido modificado, lo devolvemos */
        if(bestModelResult != null)
            return bestModelResult;

        /* En caso contrario, devolvemos un modelo "falso" con score infinitamente malo */
        return new LearningResult<>(null, bestModelScore, sem.getScoreType());
    }
}
