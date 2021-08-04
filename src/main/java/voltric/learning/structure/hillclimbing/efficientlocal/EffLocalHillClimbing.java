package voltric.learning.structure.hillclimbing.efficientlocal;

import voltric.data.DiscreteData;
import voltric.graph.Edge;
import voltric.learning.LearningResult;
import voltric.learning.score.ScoreType;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.potential.Function;
import voltric.variables.DiscreteVariable;
import voltric.variables.Variable;

import java.util.*;

/**
 * Solo tiene 3 operadores que son AddArc, DeleteArc y ReverseArc
 */
public class EffLocalHillClimbing {

    private int maxIterations;

    private double threshold;

    private Set<EffLocalHcOperator> operators;

    private ScoreType scoreType;

    public EffLocalHillClimbing(Set<EffLocalHcOperator> operators, int maxIterations, double threshold, ScoreType scoreType) {
        this.maxIterations = maxIterations;
        this.threshold = threshold;
        this.scoreType = scoreType;
        this.operators = operators;
    }

    public EffLocalHillClimbing(int maxIterations, double threshold, ScoreType scoreType) {
        this.maxIterations = maxIterations;
        this.threshold = threshold;
        this.scoreType = scoreType;
        this.operators = new LinkedHashSet<>();
        this.operators.add(new EffLocalAddArc(new ArrayList<>(), new HashMap<>(),Integer.MAX_VALUE));
        this.operators.add(new EffLocalDeleteArc(new ArrayList<>(), new HashMap<>()));
        this.operators.add(new EffLocalReverseArc(new ArrayList<>(), new HashMap<>(), Integer.MAX_VALUE));
    }

    public LearningResult<DiscreteBayesNet> learnModel(DiscreteBayesNet seedNet, DiscreteData data) {

        DiscreteBayesNet clonedNet = seedNet.clone();

        EfficientDiscreteData efficientData = new EfficientDiscreteData(data);

        /** Generate node scores for current BN */
        Map<DiscreteVariable, Double> scores = EffLocalScore.computeNetScores(clonedNet, data, this.scoreType);
        double currentScore = scores.values().stream().mapToDouble(Double::doubleValue).sum();

        int iterations = 0;
        while(iterations < this.maxIterations) {
            iterations = iterations + 1;

            /** Obtenemos la operacion optima segun el tipo de score que hemos establecido */
            EffLocalOperation optimalOperation = getOptimalOperation(clonedNet, data, efficientData, scores);

            /** Comparamos el score de la operacion optima con el score actual. Si no mejora lo suficiente cortamos el bucle. */
            if(currentScore >= optimalOperation.getScore() || Math.abs(optimalOperation.getScore() - currentScore) < threshold)
                return new LearningResult<>(clonedNet, currentScore, this.scoreType);

            /** Aplicamos la operacion optima (actualizando asi los scores intermedios) y actualizamos el score general */
            performOperation(clonedNet, data, optimalOperation, scores);
            currentScore = optimalOperation.getScore();
        }

        return new LearningResult<>(clonedNet, currentScore, this.scoreType);
    }

    private EffLocalOperation getOptimalOperation(DiscreteBayesNet bayesNet, DiscreteData data, EfficientDiscreteData efficientData, Map<DiscreteVariable, Double> scores) {

        EffLocalOperation optimalOperation = null;
        double optimalOperationScore = -Double.MAX_VALUE;

        for(EffLocalHcOperator operator: this.operators){
            EffLocalOperation localOperation = operator.apply(bayesNet, data, efficientData, scores, this.scoreType);
            if(localOperation.getScore() > optimalOperationScore) {
                optimalOperation = localOperation;
                optimalOperationScore = optimalOperation.getScore();
            }
        }

        return optimalOperation;
    }

    private void performOperation(DiscreteBayesNet bayesNet, DiscreteData data, EffLocalOperation operation, Map<DiscreteVariable, Double> scores) {
        switch(operation.getType()) {
            case OPERATION_ADD:
                /** Modifies the BN and updates the table of scores */
                performArcAddition(bayesNet, data, operation, scores);
                break;
            case OPERATION_DEL:
                /** Modifies the BN and updates the table of scores */
                performArcDeletion(bayesNet, data, operation, scores);
                break;
            case OPERATION_REVERSE:
                /** Modifies the BN and updates the table of scores */
                performArcReversal(bayesNet, data, operation, scores);
                break;
        }
    }

    private void performArcAddition(DiscreteBayesNet bayesNet, DiscreteData data, EffLocalOperation operation, Map<DiscreteVariable, Double> scores) {

        DiscreteVariable tailVar = operation.getTailVar();
        DiscreteVariable headVar = operation.getHeadVar();
        DiscreteBeliefNode tailNode = bayesNet.getNode(tailVar);
        DiscreteBeliefNode headNode = bayesNet.getNode(headVar);

        /** Project data using the node's new family */
        List<DiscreteVariable> newFamily = headNode.getDiscreteParentVariables();
        newFamily.add(headVar);
        newFamily.add(tailVar);
        DiscreteData projectedData = data.projectV3(newFamily);

        /** Create the new CPT for the node */
        Function newCpt = Function.createFunction(projectedData);
        newCpt.normalize(headVar);

        /** Modify the BN with the new arc */
        bayesNet.addEdge(headNode, tailNode);

        /** Modify the headNode's cpt */
        headNode.setCpt(newCpt);

        /** Update the local score */
        double newHeadScore = EffLocalScore.computeNodeScore(headVar, newFamily, newCpt, this.scoreType, projectedData);
        scores.put(headVar, newHeadScore);
    }

    private void performArcDeletion(DiscreteBayesNet bayesNet, DiscreteData data, EffLocalOperation operation, Map<DiscreteVariable, Double> scores) {

        DiscreteVariable tailVar = operation.getTailVar();
        DiscreteVariable headVar = operation.getHeadVar();
        DiscreteBeliefNode tailNode = bayesNet.getNode(tailVar);
        DiscreteBeliefNode headNode = bayesNet.getNode(headVar);

        /** Project data using the node's new family */
        List<DiscreteVariable> newFamily = headNode.getDiscreteParentVariables();
        newFamily.add(headVar);
        newFamily.remove(tailVar);
        DiscreteData projectedData = data.projectV3(newFamily);

        /** Create the new CPT for the node */
        Function newCpt = Function.createFunction(projectedData);
        newCpt.normalize(headVar);

        /** Modify the BN by removing the arc */
        Edge<Variable> arc = bayesNet.getEdge(headNode, tailNode).get();
        bayesNet.removeEdge(arc);

        /** Modify the headNode's cpt */
        headNode.setCpt(newCpt);

        /** Update the local score */
        double newHeadScore = EffLocalScore.computeNodeScore(headVar, newFamily, newCpt, this.scoreType, projectedData);
        scores.put(headVar, newHeadScore);
    }

    private void performArcReversal(DiscreteBayesNet bayesNet, DiscreteData data, EffLocalOperation operation, Map<DiscreteVariable, Double> scores) {
        DiscreteVariable tailVar = operation.getTailVar();
        DiscreteVariable headVar = operation.getHeadVar();
        DiscreteBeliefNode tailNode = bayesNet.getNode(tailVar);
        DiscreteBeliefNode headNode = bayesNet.getNode(headVar);

        /** Porject data using the node's new family for each case */

        List<DiscreteVariable> newTailFamily = tailNode.getDiscreteParentVariables();
        newTailFamily.add(tailVar);
        newTailFamily.add(headVar); // Add the edgeHead as another parent var in the family
        DiscreteData tailProjectedData = data.projectV3(newTailFamily);

        List<DiscreteVariable> newHeadFamily = headNode.getDiscreteParentVariables();
        newHeadFamily.add(headVar);
        newHeadFamily.remove(tailVar); // Remove the edgeTail node from the family
        DiscreteData headProjectedData = data.projectV3(newHeadFamily);

        /** Create the new CPTs for the nodes */
        Function newTailCpt = Function.createFunction(tailProjectedData);
        newTailCpt.normalize(tailVar);
        Function newHeadCpt = Function.createFunction(headProjectedData);
        newHeadCpt.normalize(headVar);

        /** Modify the BN by reversing the arc */
        Edge<Variable> arc = bayesNet.getEdge(headNode, tailNode).get();
        bayesNet.removeEdge(arc);
        bayesNet.addEdge(tailNode, headNode);

        /** Modify the CPTs of the nodes in the BN */
        headNode.setCpt(newHeadCpt);
        tailNode.setCpt(newTailCpt);

        /** Update local scores */
        double newTailScore = EffLocalScore.computeNodeScore(tailVar, newTailFamily, newTailCpt, scoreType, tailProjectedData);
        double newHeadScore = EffLocalScore.computeNodeScore(headVar, newHeadFamily, newHeadCpt, scoreType, headProjectedData);
        scores.put(headVar, newHeadScore);
        scores.put(tailVar, newTailScore);
    }
}
