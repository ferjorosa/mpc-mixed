package voltric.learning.structure.constraintbased;

import org.apache.commons.math3.util.Combinations;
import voltric.data.DiscreteData;
import voltric.graph.AbstractNode;
import voltric.graph.GraphUtilsVoltric;
import voltric.graph.UndirectedGraph;
import voltric.graph.UndirectedNode;
import voltric.learning.structure.constraintbased.test.CITest;
import voltric.util.Tuple;
import voltric.variables.DiscreteVariable;

import java.util.*;

/**
 * Created by equipo on 17/10/2017.
 */
public class SharedCbLearner implements CBLearner {

    /** Maximum number of parents per node*/
    private int maxParentOrder;

    /** P value for the independence test*/
    private double p_value;

    /** The conditional independence test that is going to be used */
    private CITest ciTest;

    /** Shared collection of c.i test scores */
    private Map<Map<Tuple<DiscreteVariable,DiscreteVariable>, List<DiscreteVariable>>, Double> ciTestScores;

    public SharedCbLearner(int maxParents,
                           double p_value,
                           CITest ciTest,
                           Map<Map<Tuple<DiscreteVariable,DiscreteVariable>, List<DiscreteVariable>>, Double> concurrentCiTestScores){
        this.maxParentOrder = maxParents;
        this.p_value = p_value;
        this.ciTest = ciTest;
        this.ciTestScores = concurrentCiTestScores;
    }

    public int getMaxParentOrder() {
        return maxParentOrder;
    }

    /**
     *
     * @param data data used to calculate the conditional independence tests
     * @return the undirected graph that contains all
     */
    public UndirectedGraph<DiscreteVariable> learnSkeleton(DiscreteData data){

        // First, a complete graph is created, representing that there are no conditional independencies.
        UndirectedGraph<DiscreteVariable> graph = GraphUtilsVoltric.createCompleteGraph(data.getVariables());

        int currentOrder = 0; // current number of parents per variable

        // While the maximum number of parents hasn't been reached iterate and look for independencies in the graph
        while(currentOrder <= maxParentOrder){

            // Iterate through all the nodes...
            for(UndirectedNode<DiscreteVariable> node: graph.getUndirectedNodes()){ // x in X

                List<Tuple<AbstractNode<DiscreteVariable>, AbstractNode<DiscreteVariable>>> edgesToBeRemoved = new ArrayList<>();

                // ...and its neighbors
                Set<AbstractNode<DiscreteVariable>> neighbors = node.getNeighbors();
                for(AbstractNode<DiscreteVariable> neighbor: neighbors) { // y in Y
                    // Create a collection containing the possible conditional neighbors
                    List<AbstractNode<DiscreteVariable>> conditionalNeighbors = new ArrayList<>(neighbors); // z in Z
                    conditionalNeighbors.remove(neighbor);

                    // Now we create a set of combinations (n,k) where n >= k
                    if (conditionalNeighbors.size() >= currentOrder) {

                        // Using Apache Math, create a set of index combinations of the conditional neighbors
                        Iterator<int[]> condNeighCombinations = new Combinations(conditionalNeighbors.size(), currentOrder).iterator();

                        // Boolean variable that stores if a c.i have been found for this set of conditional neighbors
                        boolean conditionalIndecencyFound = false;

                        while (condNeighCombinations.hasNext() && !conditionalIndecencyFound) {

                            int[] combination = condNeighCombinations.next();

                            // The index is transformed into its corresponding variable
                            List<DiscreteVariable> combinationVariables = new ArrayList<>();
                            for (int i = 0; i < combination.length; i++)
                                combinationVariables.add(conditionalNeighbors.get(i).getContent()); //TODO: Esto esta mal, deberia ser "get(combination[i])"

                            // Once the combination variables hav been established, a conditional independence test needs to be made
                            // in case it hasn't been made before
                            Map<Tuple<DiscreteVariable, DiscreteVariable>, List<DiscreteVariable>> variablesMap = new HashMap<>();
                            variablesMap.put(new Tuple<>(node.getContent(), neighbor.getContent()), combinationVariables);

                            Double testValue = ciTestScores.get(variablesMap);
                            if (testValue == null) {
                                testValue = ciTest.test(node.getContent(), neighbor.getContent(), combinationVariables, data);
                                // If the moment we want to store it, it keeps not being present, it is stored
                                this.ciTestScores.putIfAbsent(variablesMap, testValue);
                            }

                            // If the test result is inferior to the threshold established by the p_value, an independency has been found
                            // therefore the edge connecting the node and its neighbor is removed.
                            if (testValue < p_value) {
                                conditionalIndecencyFound = true;
                                // The edge to be removed is stored
                                edgesToBeRemoved.add(new Tuple<>(neighbor, node));
                            }
                        }
                    }
                }// end neighbor iteration

                for(Tuple<AbstractNode<DiscreteVariable>, AbstractNode<DiscreteVariable>> undirectedEdge: edgesToBeRemoved)
                    graph.removeEdge(undirectedEdge.getFirst(), undirectedEdge.getSecond());

            }// end node iteration

            currentOrder = currentOrder + 1;
        }

        return graph;
    }
}
