package voltric.learning.structure.hillclimbing.efficientlocal;

import voltric.data.DiscreteData;
import voltric.learning.score.ScoreType;
import voltric.model.DiscreteBayesNet;
import voltric.variables.DiscreteVariable;

import java.util.Map;

public interface EffLocalHcOperator {

    // Devuelve la mejor operacion que puede realizar dicho operador para el dataSet y en la red especifica
    EffLocalOperation apply(DiscreteBayesNet seedNet, DiscreteData data, EfficientDiscreteData efficientData, Map<DiscreteVariable, Double> scores, ScoreType scoreType);
}
