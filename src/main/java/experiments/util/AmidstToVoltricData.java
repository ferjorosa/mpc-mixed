package experiments.util;

import eu.amidst.core.datastream.Attribute;
import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.variables.stateSpaceTypes.FiniteStateSpace;
import voltric.data.DiscreteData;
import voltric.data.DiscreteDataInstance;
import voltric.variables.DiscreteVariable;
import voltric.variables.modelTypes.VariableType;

import java.util.ArrayList;
import java.util.List;

public class AmidstToVoltricData {

    public static DiscreteData transform(DataOnMemory<DataInstance> data) {

        List<DiscreteVariable> voltricVariables = new ArrayList<>();
        for(Attribute attribute: data.getAttributes()){
            FiniteStateSpace type = attribute.getStateSpaceType();
            voltricVariables.add(new DiscreteVariable(attribute.getName(), type.getStatesNames(), VariableType.MANIFEST_VARIABLE));
        }

        DiscreteData voltricData = new DiscreteData(voltricVariables);
        for(DataInstance instance: data){

            int[] instanceInt = new int[instance.getAttributes().getNumberOfAttributes()];
            for(int i = 0; i < instanceInt.length; i++)
                instanceInt[i] = (int) instance.toArray()[i];

            voltricData.add(new DiscreteDataInstance(instanceInt));
        }

        return voltricData;
    }
}
