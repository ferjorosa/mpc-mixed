package voltric.learning.structure.hillclimbing.local;

import voltric.data.DiscreteData;
import voltric.graph.DirectedAcyclicGraph;
import voltric.graph.Edge;
import voltric.learning.LearningResult;
import voltric.learning.score.ScoreType;
import voltric.learning.structure.type.StructureType;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.potential.Function;
import voltric.variables.DiscreteVariable;
import voltric.variables.Variable;

import java.util.*;

/**
 * Solo tiene 3 operadores que son AddArc, DeleteArc y ReverseArc
 */
/* TODO: Cambios necesarios:
 * Revisando el código con David nos hemos fijado en que no se esta cacheando lo suficiente, por ejemplo al añadir un
 * nuevo arco esta comprobando arcos que ya habia calculado antes. Si bien es verdad que evalua de forma local, se puede mejorar.
 *
 * Seria necesario crear dos estructuras de datos, una para almacenar los deltas de añadir un arco y otra para almacenar
 * los deltas de eliminar un arco. Para las reversiones se pueden utilizar los datos de ambas cacheadas. Las estructuras
 * serian: Una matriz para AddArc y una lista de HashMaps para DeleteArc.
 */
public class LocalHillClimbing {

    private int maxIterations;

    private double threshold;

    private Set<LocalHcOperator> operators;

    private ScoreType scoreType;

    private StructureType structureType;

    public LocalHillClimbing(Set<LocalHcOperator> operators, int maxIterations, double threshold, ScoreType scoreType, StructureType structureType) {
        this.maxIterations = maxIterations;
        this.threshold = threshold;
        this.scoreType = scoreType;
        this.operators = operators;
        this.structureType = structureType;
    }

    public LocalHillClimbing(int maxIterations, double threshold, ScoreType scoreType, StructureType structureType) {
        this.maxIterations = maxIterations;
        this.threshold = threshold;
        this.scoreType = scoreType;
        this.operators = new LinkedHashSet<>();
        this.operators.add(new LocalAddArc(new ArrayList<>(), new HashMap<>(), Integer.MAX_VALUE));
        this.operators.add(new LocalDeleteArc(new ArrayList<>(), new HashMap<>()));
        this.operators.add(new LocalReverseArc(new ArrayList<>(), new HashMap<>(), Integer.MAX_VALUE));
        this.structureType = structureType;
    }

    public LearningResult<DiscreteBayesNet> learnModel(DiscreteBayesNet seedNet, DiscreteData data) {

        /** First we clone the seedNet so it is not modified during the HC process */
        DiscreteBayesNet clonedNet = seedNet.clone();

        /** Then we create a copy graph that will be used to test the resulting local operations of each operator */
        DirectedAcyclicGraph<Variable> copyDag = generateCopyDag(clonedNet);

        EfficientDiscreteData efficientData = new EfficientDiscreteData(data);

        /** Generate node scores for current BN */
        Map<DiscreteVariable, Double> scores = LocalScore.computeNetScores(clonedNet, data, this.scoreType);
        double currentScore = scores.values().stream().mapToDouble(Double::doubleValue).sum();

        int iterations = 0;
        while(iterations < this.maxIterations) {
            iterations = iterations + 1;

            /** Obtenemos la operacion optima segun el tipo de score que hemos establecido */
            //TODO: El caso en que la optimalOperation sea null es que no se puede realizar ninguna opcion, por ejemplo que haya muchas restricciones
            LocalOperation optimalOperation = getOptimalOperation(clonedNet, data, efficientData, scores, copyDag);

            /** Comparamos el score de la operacion optima con el score actual. Si no mejora lo suficiente cortamos el bucle. */
            if (optimalOperation == null || currentScore >= optimalOperation.getScore() || Math.abs(optimalOperation.getScore() - currentScore) < threshold)
                return new LearningResult<>(clonedNet, currentScore, this.scoreType);

            /** Aplicamos la operacion optima (actualizando asi los scores intermedios) y actualizamos el score general*/
            performOperation(clonedNet, data, optimalOperation, scores);
            currentScore = optimalOperation.getScore();

            /** Actualizamos el copyDag con la operacion óptima */
            modifyCopyDag(copyDag, optimalOperation);
        }

        return new LearningResult<>(clonedNet, currentScore, this.scoreType);
    }

    private DirectedAcyclicGraph<Variable> generateCopyDag(DiscreteBayesNet net) {

        DirectedAcyclicGraph<Variable> copyDag = new DirectedAcyclicGraph<>();

        // Primero añadimos todos los nodos de la BN en forma de variables
        for(DiscreteVariable var: net.getVariables())
            copyDag.addNode(var);

        // Despues añadimos todos los edges entre dichas variables
        for(Edge<Variable> edge: net.getEdges())
            copyDag.addEdge(copyDag.getNode(edge.getHead().getContent()), copyDag.getNode(edge.getTail().getContent()));

        return copyDag;
    }

    private LocalOperation getOptimalOperation(DiscreteBayesNet bayesNet,
                                               DiscreteData data,
                                               EfficientDiscreteData efficientData,
                                               Map<DiscreteVariable, Double> scores,
                                               DirectedAcyclicGraph<Variable> copyDag) {

        LocalOperation optimalOperation = null;
        double optimalOperationScore = -Double.MAX_VALUE;

        for(LocalHcOperator operator: this.operators){

            /** Almacenamos el set de operaciones resultado del operador en cuestion */
            List<LocalOperation> localOperations = operator.apply(bayesNet, data, efficientData, scores, this.scoreType);

            /** Selecionamos la mejor operacion de la lista que cumple con los requisitos de estructura establecidos */
            if(localOperations.size() > 0) {
                LocalOperation bestOperation = selectBestOperation(localOperations, copyDag);

                /** Si la operacion mejora el current best score, se almacena como la operacion optima hasta el momento */
                if (bestOperation.getScore() > optimalOperationScore) {
                    optimalOperation = bestOperation;
                    optimalOperationScore = optimalOperation.getScore();
                }
            }
        }
        return optimalOperation;
    }

    private LocalOperation selectBestOperation(List<LocalOperation> localOperations, DirectedAcyclicGraph<Variable> copyDag) {

        /** Ordenamos la lista segun su score */
        Comparator<LocalOperation> byScore = (operation1, operation2) -> Double.compare(operation1.getScore(), operation2.getScore());
        Collections.sort(localOperations, Collections.reverseOrder(byScore));

        /** Iteramos por la lista y devolvemos la primera operacion que cumpla con las restricciones estructurales */
        for(LocalOperation operation: localOperations){
            modifyCopyDag(copyDag, operation);
            if(this.structureType.allows(copyDag)) {
                revertModifyCodpyDag(copyDag, operation);
                return operation;
            }
            else
                revertModifyCodpyDag(copyDag, operation);
        }

        /** Caso extremos donde ninguna operacion esta permitida, en tal caso, devolvemos una falsa con el peor score posible */
        return new LocalOperation(null, null, -Double.MAX_VALUE, localOperations.get(0).getType());
    }

    private void modifyCopyDag(DirectedAcyclicGraph<Variable> copyDag, LocalOperation operation) {
        switch(operation.getType()) {
            case OPERATION_ADD:
                copyDag.addEdge(copyDag.getNode(operation.getHeadVar()), copyDag.getNode(operation.getTailVar()));
                break;
            case OPERATION_DEL:
                copyDag.removeEdge(copyDag.getNode(operation.getHeadVar()), copyDag.getNode(operation.getTailVar()));
                break;
            case OPERATION_REVERSE:
                copyDag.removeEdge(copyDag.getNode(operation.getHeadVar()), copyDag.getNode(operation.getTailVar()));
                copyDag.addEdge(copyDag.getNode(operation.getTailVar()), copyDag.getNode(operation.getHeadVar()));
                break;
        }
    }

    private void revertModifyCodpyDag(DirectedAcyclicGraph<Variable> copyDag, LocalOperation operation) {
        switch(operation.getType()) {
            case OPERATION_ADD:
                copyDag.removeEdge(copyDag.getNode(operation.getHeadVar()), copyDag.getNode(operation.getTailVar()));
                break;
            case OPERATION_DEL:
                copyDag.addEdge(copyDag.getNode(operation.getHeadVar()), copyDag.getNode(operation.getTailVar()));
                break;
            case OPERATION_REVERSE:
                copyDag.removeEdge(copyDag.getNode(operation.getTailVar()), copyDag.getNode(operation.getHeadVar()));
                copyDag.addEdge(copyDag.getNode(operation.getHeadVar()), copyDag.getNode(operation.getTailVar()));
                break;
        }
    }

    private void performOperation(DiscreteBayesNet bayesNet, DiscreteData data, LocalOperation operation, Map<DiscreteVariable, Double> scores) {
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

    private void performArcAddition(DiscreteBayesNet bayesNet, DiscreteData data, LocalOperation operation, Map<DiscreteVariable, Double> scores) {

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
        Function newCpt = null;
        try {
            newCpt = Function.createFunction(projectedData);
        }catch (ArrayIndexOutOfBoundsException e) {
            throw e;
        }
        newCpt.normalize(headVar);

        /** Modify the BN with the new arc */
        bayesNet.addEdge(headNode, tailNode);

        /** Modify the headNode's cpt */
        headNode.setCpt(newCpt);

        /** Update the local score */
        double newHeadScore = LocalScore.computeNodeScore(headVar, newFamily, newCpt, this.scoreType, projectedData);
        scores.put(headVar, newHeadScore);
    }

    private void performArcDeletion(DiscreteBayesNet bayesNet, DiscreteData data, LocalOperation operation, Map<DiscreteVariable, Double> scores) {

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
        double newHeadScore = LocalScore.computeNodeScore(headVar, newFamily, newCpt, this.scoreType, projectedData);
        scores.put(headVar, newHeadScore);
    }

    private void performArcReversal(DiscreteBayesNet bayesNet, DiscreteData data, LocalOperation operation, Map<DiscreteVariable, Double> scores) {

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
        double newTailScore = LocalScore.computeNodeScore(tailVar, newTailFamily, newTailCpt, scoreType, tailProjectedData);
        double newHeadScore = LocalScore.computeNodeScore(headVar, newHeadFamily, newHeadCpt, scoreType, headProjectedData);
        scores.put(headVar, newHeadScore);
        scores.put(tailVar, newTailScore);
    }

    // No se puede añadir un arco MV -> LV
    public void newAddArcRestriction(DiscreteVariable fromVariable, DiscreteVariable toVariable) {
        for(LocalHcOperator operator: this.operators){
            if(operator instanceof LocalAddArc) {
                LocalAddArc addArcOperator = (LocalAddArc) operator;
                addArcOperator.addEdgeToBlackList(fromVariable, toVariable);
            }
        }
    }

    // No se puede revertir un arco LV -> MV
    public void newReverseArcRestriction(DiscreteVariable fromVariable, DiscreteVariable toVariable) {
        for(LocalHcOperator operator: this.operators){
            if(operator instanceof LocalReverseArc) {
                LocalReverseArc reverseArcOperator = (LocalReverseArc) operator;
                reverseArcOperator.addEdgeToBlackList(fromVariable, toVariable);
            }
        }
    }

    // Se ha eliminado del modelo dicha variable y por tanto no es necesario almacenar todas las restricciones a añadir
    // arcos desde ciertas variables a la misma
    public void removeAddArcRestrictions(DiscreteVariable toVariable) {
        for(LocalHcOperator operator: this.operators){
            if(operator instanceof LocalAddArc) {
                LocalAddArc addArcOperator = (LocalAddArc) operator;
                addArcOperator.removeNodeFromEdgeBlackList(toVariable);
            }
        }
    }

    // Se ha eliminado del modelo dicha variable y por tanto no es necesario almacenar todas las restricciones a revertir
    // arcos provenientes de la misma
    public void removeReverseArcRestrictions(DiscreteVariable fromVariable) {
        for(LocalHcOperator operator: this.operators){
            if(operator instanceof LocalReverseArc) {
                LocalReverseArc reverseArcOperator = (LocalReverseArc) operator;
                reverseArcOperator.removeNodeFromEdgeBlackList(fromVariable);
            }
        }
    }

    // TODO: Rehacer para que no tengas que poner instanceOf, una forma es no utilizar una lista de operadores, sino 3 separados, aunque limitas la extension
    // TODO: se podria hacer facilmente con polimorfismo, pero no me gustaba, asi que de momento lo hago a lo bruto
    public void copyArcRestrictions(DiscreteVariable existingVariable, DiscreteVariable newVariable) {
        for(LocalHcOperator operator: this.operators){
            if(operator instanceof LocalAddArc) {
                LocalAddArc addArcOperator = (LocalAddArc) operator;
                addArcOperator.copyArcRestrictions(existingVariable, newVariable);
            } else if(operator instanceof LocalReverseArc) {
                LocalReverseArc reverseArcOperator = (LocalReverseArc) operator;
                reverseArcOperator.copyArcRestrictions(existingVariable, newVariable);
            } else if(operator instanceof LocalDeleteArc) {
                LocalDeleteArc deleteArcOperator = (LocalDeleteArc) operator;
                deleteArcOperator.copyArcRestrictions(existingVariable, newVariable);
            }
        }
    }
}
