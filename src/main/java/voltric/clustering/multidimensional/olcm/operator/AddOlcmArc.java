package voltric.clustering.multidimensional.olcm.operator;

import voltric.data.DiscreteData;
import voltric.graph.Edge;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.variables.DiscreteVariable;
import voltric.variables.Variable;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by equipo on 16/04/2018.
 *
 * Adds a non-repeated arc between a LV and a MV, forming a OLCM
 */
public class AddOlcmArc implements OlcmHcOperator{

    private List<DiscreteVariable> manifestNodeBlackList;

    public AddOlcmArc() {
        this(new ArrayList<>());
    }

    // No se permiten arcos a los nodos de la blackList
    public AddOlcmArc(List<DiscreteVariable> manifestNodeBlackList) {
        this.manifestNodeBlackList = manifestNodeBlackList;
    }

    @Override
    public LearningResult<DiscreteBayesNet> apply(DiscreteBayesNet seedNet, DiscreteData data, AbstractEM em) {

        // The BN is copied to avoid modifying current object.
        DiscreteBayesNet clonedNet = seedNet.clone();

        double bestModelScore = -Double.MAX_VALUE; // Log-likelihood related scores are negative
        LearningResult<DiscreteBayesNet> bestModelResult = null;

        List<DiscreteBeliefNode> whiteListManifestNodes = clonedNet.getManifestNodes()
                .stream()
                .filter(x -> !this.manifestNodeBlackList.contains(x.getVariable()))
                .collect(Collectors.toList());

        // Itera por cada variable latente y añade un arco entre ella y cada MV a la cual no tenga como hijo
        for(DiscreteBeliefNode latentNode : clonedNet.getLatentNodes()){
            // Filtramos los MVs que no sean hijos de la variable latente
            for(DiscreteBeliefNode manifestNode: whiteListManifestNodes.stream()
                    .filter(x->!latentNode.getChildrenNodes().contains(x))
                    .collect(Collectors.toList())){

                // Añade un arco desde la LV a la MV no-hija
                Edge<Variable> newEdge = clonedNet.addEdge(manifestNode, latentNode);
                // Aprende la nueva red con el EM
                LearningResult<DiscreteBayesNet> newEdgeResult = em.learnModel(clonedNet, data);

                if(newEdgeResult.getScoreValue() > bestModelScore){
                    bestModelResult = newEdgeResult;
                    bestModelScore = newEdgeResult.getScoreValue();
                }

                // Independientemente de si es mejor o no, eliminamos el nuevo arco para dejarlo como la estructura inicial
                clonedNet.removeEdge(newEdge);
            }
        }

        if(bestModelResult != null)
            return bestModelResult;

        //throw new IllegalStateException("No me creo que no haya ningun modelo que mejore el -double.MAX_VALUE score");
        return new LearningResult<>(null, bestModelScore, em.getScoreType());
    }
}
