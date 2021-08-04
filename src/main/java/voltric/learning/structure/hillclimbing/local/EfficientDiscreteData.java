package voltric.learning.structure.hillclimbing.local;

import voltric.data.DiscreteData;
import voltric.data.DiscreteDataInstance;

//TODO: Es unicamente temporal, para ver si con este approach consigo mejorar los tiempos de calculo del LocalHillclimbing
public class EfficientDiscreteData {

    int[][] content;

    int[] weights;

    int totalWeight;

    public EfficientDiscreteData(DiscreteData data) {
        this.totalWeight = data.getTotalWeight();

        this.weights = new int[data.getInstances().size()];
        this.content = new int[data.getInstances().size()][data.getVariables().size()];

        for(int i=0; i < data.getInstances().size(); i++){
            DiscreteDataInstance dataInstance = data.getInstances().get(i);

            for(int j = 0; j < dataInstance.size(); j++)
                this.content[i][j] = dataInstance.getNumericValue(j);

            this.weights[i] = data.getWeight(dataInstance);
        }
    }
}
