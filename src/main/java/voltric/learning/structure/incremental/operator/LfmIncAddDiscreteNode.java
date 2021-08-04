package voltric.learning.structure.incremental.operator;

import eu.amidst.extension.util.tuple.Tuple3;
import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.LocalEM;
import voltric.learning.parameter.em.config.EmConfig;
import voltric.learning.structure.incremental.localemtype.TypeLocalEM;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.variables.DiscreteVariable;
import voltric.variables.modelTypes.VariableType;

import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;

/* Dado que utilizamos LocalEM, tenemos que crear nuevas instancias o resetearlas con cada llamada al operador */
public class LfmIncAddDiscreteNode implements LfmIncOperator {

    private int newNodeCardinality;

    private int maxNumberOfDiscreteLatentNodes;

    private EmConfig emConfig;

    private TypeLocalEM typeLocalEM;

    private int latentVarNameCounter = 1;

    public LfmIncAddDiscreteNode(int newNodeCardinality,
                                 int maxNumberOfDiscreteLatentNodes,
                                 EmConfig emConfig,
                                 TypeLocalEM typeLocalEM) {
        this.newNodeCardinality = newNodeCardinality;
        this.maxNumberOfDiscreteLatentNodes = maxNumberOfDiscreteLatentNodes;
        this.emConfig = emConfig;
        this.typeLocalEM = typeLocalEM;
    }

    @Override
    public Tuple3<DiscreteVariable, DiscreteVariable, LearningResult<DiscreteBayesNet>> apply(List<String> currentset,
                                                                                              DiscreteBayesNet bayesNet,
                                                                                              DiscreteData data){

        DiscreteVariable bestFirstVar = null;
        DiscreteVariable bestSecondVar = null;
        double bestModelScore = -Double.MAX_VALUE;
        LearningResult<DiscreteBayesNet> bestModelResult = null;

        /* Return current model if current number of discrete latent nodes is maximum */
        int numberOfLatentNodes = bayesNet.getLatentVariables().size();
        if(numberOfLatentNodes >= this.maxNumberOfDiscreteLatentNodes)
            return new Tuple3<>(bestFirstVar, bestSecondVar, new LearningResult<>(bayesNet, bestModelScore, emConfig.getScoreType()));

        /* The BN is copied to avoid modifying current object */
        DiscreteBayesNet clonedNet = bayesNet.clone();

        /* Define the set of variables */
        List<DiscreteVariable> currentSetVariables = new ArrayList<>(currentset.size());
        for(String varName: currentset)
            currentSetVariables.add(clonedNet.getNode(varName).getVariable());

        for(DiscreteVariable firstVar: currentSetVariables) {
            for (DiscreteVariable secondVar : currentSetVariables) {
                if(!firstVar.equals(secondVar)){

                    /* Create a new LV as the pair's new parent */
                    DiscreteVariable newLatentVar = new DiscreteVariable(this.newNodeCardinality, VariableType.LATENT_VARIABLE, "LV_" + this.latentVarNameCounter);
                    DiscreteBeliefNode newLatentNode = clonedNet.addNode(newLatentVar);
                    clonedNet.addEdge(clonedNet.getNode(firstVar), newLatentNode);
                    clonedNet.addEdge(clonedNet.getNode(secondVar), newLatentNode);

                    /* Locally learn the parameters of the new model (consider the new LV's markov blanket) */
                    Set<DiscreteVariable> localVariables = typeLocalEM.variablesToUpdate(newLatentNode);
                    LocalEM localEM = new LocalEM(localVariables, emConfig);
                    LearningResult<DiscreteBayesNet> newLatentVarResult = localEM.learnModel(clonedNet, data);

                    /* Store the new model if score improves */
                    if (newLatentVarResult.getScoreValue() > bestModelScore) {
                        bestModelScore = newLatentVarResult.getScoreValue();
                        bestModelResult = newLatentVarResult;
                        bestFirstVar = firstVar;
                        bestSecondVar = secondVar;
                    }

                    /* Independently, original model structure is restored */
                    clonedNet.removeNode(newLatentNode);
                }
            }
        }

        if(bestModelResult != null) {
            this.latentVarNameCounter++;
            bestModelResult.setName("AddDiscreteNode");
            return new Tuple3<>(bestFirstVar, bestSecondVar, bestModelResult);
        }

        /* This case implies that no model have improved the -Double.MAX_VALUE score */
        return new Tuple3<>(bestFirstVar, bestSecondVar, new LearningResult<>(bayesNet, bestModelScore, emConfig.getScoreType(), "AddDiscreteNode"));
    }

    @Override
    public Tuple3<DiscreteVariable, DiscreteVariable, LearningResult<DiscreteBayesNet>> apply(PriorityQueue<Tuple3<String, String, Double>> selectedTriples, DiscreteBayesNet bayesNet, DiscreteData data) {

        DiscreteVariable bestFirstVar = null;
        DiscreteVariable bestSecondVar = null;
        double bestModelScore = -Double.MAX_VALUE;
        LearningResult<DiscreteBayesNet> bestModelResult = null;

        /* The BN is copied to avoid modifying current object */
        DiscreteBayesNet clonedNet = bayesNet.clone();

        /* Iterate through the selected triples */
        for(Tuple3<String, String, Double> triple: selectedTriples){
            DiscreteVariable firstVar = clonedNet.getNode(triple.getFirst()).getVariable();
            DiscreteVariable secondVar = clonedNet.getNode(triple.getSecond()).getVariable();

            /* Create a new LV as the pair's new parent */
            DiscreteVariable newLatentVar = new DiscreteVariable(this.newNodeCardinality, VariableType.LATENT_VARIABLE, "LV_" + this.latentVarNameCounter);
            DiscreteBeliefNode newLatentNode = clonedNet.addNode(newLatentVar);
            clonedNet.addEdge(clonedNet.getNode(firstVar), newLatentNode);
            clonedNet.addEdge(clonedNet.getNode(secondVar), newLatentNode);

            /* Locally learn the parameters of the new model (consider the new LV's markov blanket) */
            Set<DiscreteVariable> localVariables = typeLocalEM.variablesToUpdate(newLatentNode);
            LocalEM localEM = new LocalEM(localVariables, emConfig);
            LearningResult<DiscreteBayesNet> newLatentVarResult = localEM.learnModel(clonedNet, data);

            /* Store the new model if score improves */
            if (newLatentVarResult.getScoreValue() > bestModelScore) {
                bestModelScore = newLatentVarResult.getScoreValue();
                bestModelResult = newLatentVarResult;
                bestFirstVar = firstVar;
                bestSecondVar = secondVar;
            }

            /* Independently, original model structure is restored */
            clonedNet.removeNode(newLatentNode);
        }

        if(bestModelResult != null) {
            this.latentVarNameCounter++;
            bestModelResult.setName("AddDiscreteNode");
            return new Tuple3<>(bestFirstVar, bestSecondVar, bestModelResult);
        }

        /* This case implies that no model have improved the -Double.MAX_VALUE score */
        return new Tuple3<>(bestFirstVar, bestSecondVar, new LearningResult<>(bayesNet, bestModelScore, emConfig.getScoreType(), "AddDiscreteNode"));
    }
}
