package voltric.clustering.multidimensional.olcm.operator;

import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.variables.DiscreteVariable;
import voltric.variables.Variable;
import voltric.variables.modelTypes.VariableType;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by equipo on 16/04/2018.
 *
 * Añade un nodo latente entre el par de MVs que genera un score mas alto
 */
public class OldAddOlcmNode implements OlcmHcOperator {

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

                    // Primero comprobamos que no exista un nodo cuyos 2 unicos hijos sean "firstMV" y "secondMV"
                    List<DiscreteBeliefNode> repeatedLatentNodes = clonedNet.getLatentNodes().stream()
                            .filter(x -> {
                                List<Variable> childrenVars = x.getChildrenNodes().stream().map(y -> y.getVariable()).collect(Collectors.toList());
                                return childrenVars.size() == 2 && childrenVars.contains(firstMV) && childrenVars.contains(secondMV);
                            })
                            .collect(Collectors.toList());
                    if (repeatedLatentNodes.size() == 0) {
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
