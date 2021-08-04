package voltric.clustering.multidimensional.olcm.operator;

import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.variables.DiscreteVariable;
import voltric.variables.Variable;
import voltric.variables.modelTypes.VariableType;

import java.util.Set;
import java.util.stream.Collectors;

/**
 * Created by equipo on 16/04/2018.
 *
 * Añade un nodo latente entre el par de MVs que genera un score mas alto
 *
 * TODO: Segun la version del 23-04-2017, solo puedo añadir una LV entre 2 MVs que no pertenezcan a la misma particion. Lo hago para reducir el numero de posibilidades
 */
public class AddOlcmNode implements OlcmHcOperator {

    @Override
    public LearningResult<DiscreteBayesNet> apply(DiscreteBayesNet seedNet, DiscreteData data, AbstractEM em) {

        // The BN is copied to avoid modifying current object.
        DiscreteBayesNet clonedNet = seedNet.clone();

        double bestModelScore = -Double.MAX_VALUE;
        LearningResult<DiscreteBayesNet> bestModelResult = null;

        // Por cada par de MVs se añade una LV siempre y cuando no exista ya dicha LV especificamente
        for(DiscreteVariable firstMV: clonedNet.getManifestVariables()) {
            for (DiscreteVariable secondMV : clonedNet.getManifestVariables()) {
                if (!firstMV.equals(secondMV)) {
                    // Comprobamos que ambos nodos no tengan un padre LV en común

                    // Cogemos el set de padres de first, si el set de padres de second contiene alguno del primero, return false
                    Set<Variable> firstParents = clonedNet.getNode(firstMV).getParents().stream().map(x-> x.getContent()).collect(Collectors.toSet());
                    Set<Variable> secondParents = clonedNet.getNode(secondMV).getParents().stream().map(x-> x.getContent()).collect(Collectors.toSet());
                    boolean parentInCommon = false;

                    for(Variable firstParent: firstParents)
                        if(secondParents.contains(firstParent))
                            parentInCommon = true;

                    if (!parentInCommon) {
                        // Creamos un nodo latente cuyos 2 hijos son "firstMV" y "secondMV"
                        DiscreteVariable newLatentVar = new DiscreteVariable(2, VariableType.LATENT_VARIABLE);
                        DiscreteBeliefNode newLatentNode = clonedNet.addNode(newLatentVar);
                        clonedNet.addEdge(clonedNet.getNode(firstMV), newLatentNode);
                        clonedNet.addEdge(clonedNet.getNode(secondMV), newLatentNode);

                        // Aprendemos los parametros de este nuevo modelo
                        LearningResult<DiscreteBayesNet> newLatentVarResult = em.learnModel(clonedNet, data);

                        // Si el score del modelo al añadir esta variable que representa la relacion
                        if (newLatentVarResult.getScoreValue() > bestModelScore) {
                            bestModelScore = newLatentVarResult.getScoreValue();
                            bestModelResult = newLatentVarResult;
                        }

                        // Independientemente de si el nuevo nodo ha mejorado el score o no, lo eliminamos al terminar la iteracion
                        // y con el los arcos asociados
                        clonedNet.removeNode(newLatentNode);
                    }
                }
            }
        }

        if(bestModelResult != null)
            return bestModelResult;

        //throw new IllegalStateException("No me creo que no haya ningun modelo que mejore el -double.MAX_VALUE score");
        return new LearningResult<>(null, bestModelScore, em.getScoreType());
    }
}
