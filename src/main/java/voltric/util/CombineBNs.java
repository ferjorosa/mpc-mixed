package voltric.util;

import voltric.data.DiscreteData;
import voltric.graph.Edge;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.potential.Function;
import voltric.variables.DiscreteVariable;
import voltric.variables.Variable;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Combina varias BNs en una sola.
 *
 * NOTA: Por ahora es necesario pasarle el DataSet ya que asi le asigna el indice adecuado a las variables manifest
 */
public class CombineBNs {

    /** Combines two BNs into a single one */
    public static DiscreteBayesNet combine(DiscreteBayesNet bn1, DiscreteBayesNet bn2, DiscreteData data) {

       List<DiscreteBayesNet> bns = new ArrayList<>();
       bns.add(bn1);
       bns.add(bn2);

       return combine(bns, data);
    }

    public static DiscreteBayesNet combine(List<DiscreteBayesNet> bns, DiscreteData data) {

        /* 1 - Create the new BN */
        DiscreteBayesNet combinedBn = new DiscreteBayesNet("combined_BN");

        /* 2 - Add all manifest variables */
        for(DiscreteBayesNet bn: bns)
            for(DiscreteVariable mv: bn.getManifestVariables()) {

                // Select the corresponding dataVar to the manifest var in the model
                Optional<DiscreteVariable> dataVar = data.getVariable(mv.getName());
                if(!dataVar.isPresent())
                    throw new IllegalArgumentException("All the manifest variables in the combining models must be on the data: " + mv.getName());

                // Avoid adding repeated nodes or an exception will be thrown
                if (!combinedBn.containsVar(dataVar.get()))
                    combinedBn.addNode(dataVar.get());
            }

        /* 3 - Add all latent variables (create new ones with new indexes so the will be last)*/
        for(DiscreteBayesNet bn: bns)
            for(DiscreteVariable lv: bn.getLatentVariables())
                combinedBn.addNode(new DiscreteVariable(lv.getName(), lv.getStates(), lv.getType()));


        /* 4 - Add all the edges from the old BNs */
        for(DiscreteBayesNet bn: bns)
            for(Edge<Variable> edge: bn.getEdges()){
                // Usamos el nombre de la variable para que no haya confusion con el index y se seleccione la adecuada
                DiscreteBeliefNode head = combinedBn.getNode(edge.getHead().getContent().getName());
                DiscreteBeliefNode tail = combinedBn.getNode(edge.getTail().getContent().getName());

                // Avoid adding repeated edges or an exception will be thrown
                if(!combinedBn.containsEdge(head, tail))
                    combinedBn.addEdge(head, tail);
            }

        /* 5 - Copy all the parameters from the old networks */
        for(DiscreteBayesNet bn: bns)
            for(DiscreteBeliefNode node: bn.getNodes()){
                DiscreteBeliefNode combinedBnNode = combinedBn.getNode(node.getName());

                // Aqui cogemos las variables del nodo de la combinedBn, pero lo demas de la CPT (sus valores) vienen de la que se copia
                Function copiedFunction = Function.createFullyDefinedFunction(combinedBnNode.getCpt().getVariables(), node.getCpt().getCells(), node.getCpt().getMagnitudes());
                combinedBnNode.setCpt(copiedFunction);
            }

        return combinedBn;
    }
}
