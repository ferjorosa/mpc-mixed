package voltric.clustering.multidimensional.olcm.operator;

import voltric.data.DiscreteData;
import voltric.graph.Edge;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.model.AbstractBeliefNode;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.variables.Variable;

/**
 * Created by equipo on 16/04/2018.
 */
public class RemoveOlcmArc implements OlcmHcOperator{

    @Override
    public LearningResult<DiscreteBayesNet> apply(DiscreteBayesNet seedNet, DiscreteData data, AbstractEM em) {
        // The BN is copied to avoid modifying current object.
        DiscreteBayesNet clonedNet = seedNet.clone();

        double bestModelScore = -Double.MAX_VALUE; // Log-likelihood related scores are negative
        LearningResult<DiscreteBayesNet> bestModelResult = null;

        // Itera por cada variable latente y elimina un arco entre ella y cada una de sus hijas MV
        for(DiscreteBeliefNode latentNode : clonedNet.getLatentNodes()){
            // Filtramos los MVs que no sean hijos de la variable latente
            for(AbstractBeliefNode manifestNode: latentNode.getChildrenNodes()){

                // Elimina el arco entre ambos
                Edge<Variable> edgeToBeRemoved = clonedNet.getEdge(manifestNode, latentNode).get();
                clonedNet.removeEdge(edgeToBeRemoved);

                // Aprende la nueva red con el EM
                LearningResult<DiscreteBayesNet> newEdgeResult = em.learnModel(clonedNet, data);

                if(newEdgeResult.getScoreValue() > bestModelScore){
                    bestModelResult = newEdgeResult;
                    bestModelScore = newEdgeResult.getScoreValue();
                }

                // Independientemente de si es mejor o no, añadimos de nuevo el arco
                clonedNet.addEdge(manifestNode, latentNode);
            }
        }

        if(bestModelResult != null)
            return bestModelResult;

        //throw new IllegalStateException("No me creo que no haya ningun modelo que mejore el -double.MAX_VALUE score");
        return new LearningResult<>(null, bestModelScore, em.getScoreType());
    }
}
