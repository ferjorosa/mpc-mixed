package voltric.learning.structure.latent;


import org.apache.commons.lang3.NotImplementedException;
import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.learning.structure.hillclimbing.local.LocalHillClimbing;
import voltric.learning.structure.type.StructureType;
import voltric.model.DiscreteBayesNet;
import voltric.variables.DiscreteVariable;

import java.util.List;
import java.util.Map;

;

/**
 * El objetivo de esta clase es, basandose en los MBCs CB-descomponibles, dividir el proceso del SEM en 2 partes:
 * - Class-Bridge graph
 * - Feature graph
 *
 * "Ventajas":
 *  - Podemos restringir la estructura de cada una de las partes para que sea diferente (i.e. PolyTree - DAG)
 *  - El LocalHillClimbing se hace en dos partes, lo que puede incrementar la eficiencia del algoritmo
 *
 *  Tiene por tanto 2 HCs, uno para CB y otro para Feature
 */
public class CB_StructuralEM {

    private List<DiscreteVariable> classVars;

    private List<DiscreteVariable> featureVars;

    private StructureType classBridgeStructure;

    private StructureType featureStructure;

    private AbstractEM em;

    private int maxIterations;

    private int maxNumberOfParents;

    private LocalHillClimbing hillClimbing;

    public CB_StructuralEM(List<DiscreteVariable> classVars,
                           List<DiscreteVariable> featureVars,
                           Map<DiscreteVariable, List<DiscreteVariable>> forbiddenDeleteArcs,
                           Map<DiscreteVariable, List<DiscreteVariable>> extraForbiddenAddArcs,
                           Map<DiscreteVariable, List<DiscreteVariable>> extraForbiddenReverseArcs,
                           StructureType classBridgeStructure,
                           StructureType featureStructure,
                           AbstractEM em,
                           int maxIterations,
                           int maxNumberOfParents) {

        this.classVars = classVars;
        this.featureVars = featureVars;
        this.classBridgeStructure = classBridgeStructure;
        this.featureStructure = featureStructure;
        this.em = em;
        this.maxIterations = maxIterations;
        this.maxNumberOfParents = maxNumberOfParents;
    }

    // el modelo seedNetResult debe de encontrarse aprendido con los parametros adecuados (EM algorithm)
    public LearningResult<DiscreteBayesNet> learnModel(LearningResult<DiscreteBayesNet> seedNetResult, DiscreteData data) {

        DiscreteBayesNet currentNet = seedNetResult.getBayesianNetwork();
        double currentScore = seedNetResult.getScoreValue();

        throw new NotImplementedException("Nope");
    }

}
