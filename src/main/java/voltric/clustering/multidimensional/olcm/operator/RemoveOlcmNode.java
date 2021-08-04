package voltric.clustering.multidimensional.olcm.operator;

import voltric.clustering.multidimensional.olcm.OlcmHillClimbing;
import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.variables.DiscreteVariable;
import voltric.variables.Variable;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Created by equipo on 30/04/2018.
 *
 * Elimina el nodo latente y añade de forma iterativa los mejores arcos desde el resto de variables latentes. En cierto
 * sentido, busca una sustitucion del modelo con X variables por el mismo modelo con (X-1)
 *
 * Independientemente de si las hijas de la variable latente eliminada poseen otros padres latentes o no, se les incluye
 * en el proceso de busqueda en el cual simplemente se intenta añadir nuevos arcos de las demas variables latentes a ellas
 * Es un proceso iterativo y greedy.
 */
public class RemoveOlcmNode implements OlcmHcOperator {

    @Override
    public LearningResult<DiscreteBayesNet> apply(DiscreteBayesNet seedNet, DiscreteData data, AbstractEM em) {

        // The BN is copied to avoid modifying current object.
        DiscreteBayesNet clonedNet = seedNet.clone();

        double bestModelScore = -Double.MAX_VALUE;
        LearningResult<DiscreteBayesNet> bestModelResult = null;

        List<DiscreteBeliefNode> copiedLatentNodes = new ArrayList<>(clonedNet.getLatentNodes());

        for(DiscreteBeliefNode latentNode: copiedLatentNodes){
            DiscreteVariable currentlyRemovedNodeVariable = latentNode.getVariable();
            List<Variable> currentlyRemovedNodeChildren = latentNode.getChildrenNodes().stream().map(x-> x.getVariable()).collect(Collectors.toList());;

            // Los hijos del Latent Node sobre los cuales se ejecutara el proceso Hill-climbing
            List<DiscreteVariable> latentNodeChildren = latentNode.getChildrenNodes().stream().map(x->(DiscreteVariable) x.getContent()).collect(Collectors.toList());

            // Despues lo eliminamos
            clonedNet.removeNode(latentNode);

            LearningResult<DiscreteBayesNet> currentVarModelResult = learnModelWithCurrentPartitions(clonedNet, data, em, latentNodeChildren);

            if(currentVarModelResult.getScoreValue() > bestModelScore) {
                bestModelResult = currentVarModelResult;
                bestModelScore = currentVarModelResult.getScoreValue();
            }

            // Como hemos hecho una clonacion en "learnWithCurrentPArtitions", simplemente reañadimos el nodo a la seedNet para reiniciar el modelo
            DiscreteBeliefNode currentlyRemovedNode = clonedNet.addNode(currentlyRemovedNodeVariable);
            for(Variable child: currentlyRemovedNodeChildren)
                clonedNet.addEdge(clonedNet.getNode(child), currentlyRemovedNode);

        }

        return bestModelResult;
    }

    private LearningResult<DiscreteBayesNet> learnModelWithCurrentPartitions(DiscreteBayesNet initialModel,
                                                                             DiscreteData data,
                                                                             AbstractEM em,
                                                                             List<DiscreteVariable> whiteListManifestNodes) {

        List<DiscreteVariable> blackListManifestVars = initialModel.getManifestVariables()
                .stream()
                .filter(x -> !whiteListManifestNodes.contains(x)).collect(Collectors.toList());

        /** Expansion Operators */
        Set<OlcmHcOperator> expansionOperators = new HashSet<>();
        expansionOperators.add(new AddOlcmArc(blackListManifestVars));

        /** Simplification Operators */
        Set<OlcmHcOperator> simplificationOperators = new HashSet<>();

        OlcmHillClimbing hillClimbing = new OlcmHillClimbing(400, 0.5, expansionOperators, simplificationOperators);

        return hillClimbing.learnModel(initialModel, data, em);
    }
}
