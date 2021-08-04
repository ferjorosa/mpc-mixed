package experiments.util;

import eu.amidst.core.distribution.ConditionalDistribution;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.ParentSet;
import eu.amidst.core.variables.StateSpaceTypeEnum;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.stateSpaceTypes.FiniteStateSpace;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.potential.Function;
import voltric.variables.DiscreteVariable;
import voltric.variables.modelTypes.VariableType;

import java.util.List;
import java.util.stream.Collectors;

public class AmidstToVoltricModel {

    /* Por ahora esta pensado para arboles, si queremos extenderlo habria que pensar mejor el orden de las variables en la CPT */
    public static DiscreteBayesNet transform(BayesianNetwork amidstLTM) {

        if(amidstLTM.getVariables().getListOfVariables().stream()
                .anyMatch(x -> x.getStateSpaceTypeEnum() == StateSpaceTypeEnum.REAL)) {
            throw new IllegalArgumentException("Only Discrete LTMs can be converted to Voltric format");
        }

        List<Variable> manifestVariables = amidstLTM.getVariables().getListOfVariables().stream()
                .filter(x-> x.getAttribute() != null).collect(Collectors.toList());

        List<Variable> latentVariables = amidstLTM.getVariables().getListOfVariables().stream()
                .filter(x-> x.getAttribute() == null).collect(Collectors.toList());

        DiscreteBayesNet voltricLTM = new DiscreteBayesNet(amidstLTM.getName());

        /* Añadimos las variables manifest al modelo */
        for(Variable manifestVariable: manifestVariables){
            FiniteStateSpace type = manifestVariable.getStateSpaceType();
            voltricLTM.addNode(new DiscreteVariable(manifestVariable.getName(), type.getStatesNames(), VariableType.MANIFEST_VARIABLE));
        }

        /* Añadimos las variables latentes al modelo */
        for(Variable latentVariable: latentVariables) {
            FiniteStateSpace type = latentVariable.getStateSpaceType();
            voltricLTM.addNode(new DiscreteVariable(latentVariable.getName(), type.getStatesNames(), VariableType.LATENT_VARIABLE));
        }

        /* Añadimos los arcos */
        for(ParentSet parentSet: amidstLTM.getDAG().getParentSets()){
            List<String> parentNames = parentSet.getParents().stream().map(x->x.getName()).collect(Collectors.toList());
            String mainVarName = parentSet.getMainVar().getName();

            for(String parentName: parentNames) {
                DiscreteBeliefNode mainNode = voltricLTM.getNode(mainVarName);
                DiscreteBeliefNode parentNode = voltricLTM.getNode(parentName);
                voltricLTM.addEdge(mainNode, parentNode);
            }
        }

        /* Modificamos las CPTs segun su valores en la red de AMIDST */
        for(ConditionalDistribution dist: amidstLTM.getConditionalDistributions()){

            /* Obtenemos la CPT correspondiente a la variable, trabajamos con ella de forma diferente si es raiz */
            DiscreteBeliefNode node = voltricLTM.getNode(dist.getVariable().getName());
            Function cpt = node.getCpt();
            int[] indices; // No es mas que un vector de dimension 1 con los estados para computeIndex

            /* Variable raiz */
            if(dist.getConditioningVariables().size() == 0 ) {
                indices = new int[1]; // Una sola variable
                for (int i = 0; i < cpt.getDomainSize(); i++) {
                    indices[0] = i;
                    cpt.getCells()[cpt.computeIndex(indices)] = dist.getParameters()[i];
                }
            /*
             * Variable interna u hoja.
             * TODO: Investigar el orden que tienen las variables en una CPT de Amidst y ajustar para Voltric.
             */
            } else {

                int[] _indices = new int[2];
                int array_index = 0;
                List<DiscreteVariable> _variables = cpt.getVariables();
                for(int i=0; i < _variables.get(0).getCardinality(); i++)
                    for(int j=0; j < _variables.get(1).getCardinality(); j++){
                        _indices[0] = i;
                        _indices[1] = j;
                        int index = cpt.computeIndex(_indices);
                        cpt.getCells()[index] = dist.getParameters()[array_index++];
                    }
            }
        }

        return voltricLTM;
    }
}
