package voltric.learning.structure.hillclimbing.local;

import voltric.data.DiscreteData;
import voltric.learning.score.ScoreType;
import voltric.model.DiscreteBayesNet;
import voltric.variables.DiscreteVariable;

import java.util.List;
import java.util.Map;

public interface LocalHcOperator {

    // Devuelve el conjunto de operaciones que no generan un ciclo en la BN
    // TODO: Puede que haya mejoras de tiempo, leer el archivo Mejoras.txt
    List<LocalOperation> apply(DiscreteBayesNet seedNet, DiscreteData data, EfficientDiscreteData efficientData, Map<DiscreteVariable, Double> scores, ScoreType scoreType);
}
