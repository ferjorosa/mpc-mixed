package voltric.util;

import voltric.data.DiscreteData;
import voltric.data.DiscreteDataInstance;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.potential.Function;
import voltric.variables.DiscreteVariable;

import java.util.*;


public class BayesNetSampler {

    public static DiscreteData sample(DiscreteBayesNet bn, int nSamples, Random random) {

        /* Generate the sampled dataset */
        DiscreteData sampledData = new DiscreteData(bn.getVariables());

        /* Sort BN nodes in topological order */
        List<DiscreteBeliefNode> sortedNodes = bn.topologicalSort();

        /* Store the index of those nodes in a Map*/
        Map<DiscreteVariable, Integer> mapVarIndex = new HashMap<>();
        for(DiscreteBeliefNode node: sortedNodes)
            mapVarIndex.put(node.getVariable(), sortedNodes.indexOf(node));

        List<DiscreteDataInstance> instances = new ArrayList<>(); // REMOVE, created just for testing

        /* 3 - Generate samples in order */
        for(int i = 0; i < nSamples; i++) {
            int[] states = new int[sortedNodes.size()];
            for(DiscreteBeliefNode node: sortedNodes){

                List<DiscreteVariable> parents = node.getDiscreteParentVariables();
                ArrayList<Integer> parentStates = new ArrayList<>();
                for(DiscreteVariable parent: parents) {
                    int pos = mapVarIndex.get(parent);
                    parentStates.add(states[pos]);
                }

                /* Instantiate parents */
                Function cond = node.getCpt().project(parents, parentStates);

                /* Sample according to the conditional distribution */
                states[mapVarIndex.get(node.getVariable())] = cond.sample(random);
            }

            DiscreteDataInstance instance = new DiscreteDataInstance(states);
            sampledData.add(instance);
            instances.add(instance);
        }

        return sampledData;
    }

    public static DiscreteData newSample(DiscreteBayesNet bn, int nSamples, Random random) {

        /* Generate the sampled dataset */
        DiscreteData sampledData = new DiscreteData(bn.getVariables());

        /* Sort BN nodes in topological order */
        List<DiscreteBeliefNode> sortedNodes = bn.topologicalSort();

        /* Store the index of the variables in a map */
        Map<DiscreteVariable, Integer> mapVarIndex = new HashMap<>();
        for(DiscreteVariable var: bn.getVariables())
            mapVarIndex.put(var, bn.getVariables().indexOf(var));

        List<DiscreteDataInstance> instances = new ArrayList<>(); // REMOVE, created just for testing

        /* 3 - Generate samples in order */
        for(int i = 0; i < nSamples; i++) {
            int[] states = new int[sortedNodes.size()];
            for(DiscreteBeliefNode node: sortedNodes){

                List<DiscreteVariable> parents = node.getDiscreteParentVariables();
                ArrayList<Integer> parentStates = new ArrayList<>();
                for(DiscreteVariable parent: parents) {
                    int pos = mapVarIndex.get(parent);
                    parentStates.add(states[pos]);
                }

                /* Instantiate parents */
                Function cond = node.getCpt().project(parents, parentStates);

                /* Sample according to the conditional distribution */
                states[mapVarIndex.get(node.getVariable())] = cond.sample(random);
            }

            DiscreteDataInstance instance = new DiscreteDataInstance(states);
            sampledData.add(instance);
            instances.add(instance);
        }

        return sampledData;
    }
}
