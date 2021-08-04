package voltric.clustering.unidimensional;

import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.parameter.DiscreteParameterLearning;
import voltric.learning.structure.hillclimbing.global.DecreaseLatentCardinality;
import voltric.learning.structure.hillclimbing.global.GlobalHillClimbing;
import voltric.learning.structure.hillclimbing.global.HcOperator;
import voltric.learning.structure.hillclimbing.global.IncreaseLatentCardinality;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.model.creator.HlcmCreator;
import voltric.util.distance.Hellinger;
import voltric.variables.DiscreteVariable;

import java.util.*;
import java.util.stream.Collectors;

public class HellingerWrapper {

    public static LearningResult<DiscreteBayesNet> learnLCM(DiscreteData data, double threshold, DiscreteParameterLearning parameterLearning, DiscreteBayesNet initialModel) {
        /* 1 - Creamos dos HillClimbers, uno para incrementar la card y otro para decrementar la cardinalidad */
        int maxCardinality = Integer.MAX_VALUE;
        IncreaseLatentCardinality ilcOperator = new IncreaseLatentCardinality(maxCardinality);
        Set<HcOperator> ilcOperatorSet = new HashSet<>();
        ilcOperatorSet.add(ilcOperator);
        GlobalHillClimbing increaseOnlyHC = new GlobalHillClimbing(ilcOperatorSet, Integer.MAX_VALUE, threshold);//No limitamos el numero de iteraciones posibles

        int minCardinality = 2;
        DecreaseLatentCardinality dlcOperator = new DecreaseLatentCardinality(minCardinality);
        Set<HcOperator> dlcOperatorSet = new HashSet<>();
        dlcOperatorSet.add(dlcOperator);
        GlobalHillClimbing decreaseOnlyHC = new GlobalHillClimbing(dlcOperatorSet, Integer.MAX_VALUE, threshold);//No limitamos el numero de iteraciones posibles

        DiscreteData currentData = data;
        DiscreteBayesNet currentModel = initialModel;

        /* 2 - Bucle infinito que seguira ejecutandose hasta que se dejen de filtrar variables */
        int iter = 0;
        while(true) {

            System.out.println("Iteracion: "+(++iter));

            /* 2.1 - Aprendemos un modelo con incrementar y con decrementar por separado y nos quedamos con el que obtenga mejor score*/
            LearningResult<DiscreteBayesNet> increaseCardModel = increaseOnlyHC.learnModel(currentModel, currentData, parameterLearning);
            LearningResult<DiscreteBayesNet> decreaseCardModel = decreaseOnlyHC.learnModel(currentModel, currentData, parameterLearning);

            System.out.println("Score del increaseCardModel: " + increaseCardModel.getScoreValue());
            System.out.println("Score del decreaseCardModel: " + decreaseCardModel.getScoreValue());

            LearningResult<DiscreteBayesNet> bestModel;

            if(increaseCardModel.getScoreValue() > decreaseCardModel.getScoreValue())
                bestModel = increaseCardModel;
            else
                bestModel = decreaseCardModel;

            /* 2.2 - Filtramos las variables del modelo */
            List<DiscreteVariable> variablesToFilter = HellingerWrapper.filterVariables(bestModel.getBayesianNetwork(),
                    bestModel.getBayesianNetwork().getLatentVariables().get(0), threshold);

            System.out.println("Number of filtered variables: "+ variablesToFilter.size());

            /* 2.2.1 - En caso de que no se filtre ninguna, devolvemos el modelo */
            if(variablesToFilter.size() == 0)
                return bestModel;

            /* 2.2.2 - Por el contrario, si hay variables filtradas, creamos un nuevo modelo y projectamos los datos con la nueva dimensionalidad */
            DiscreteBayesNet filteredModel = new DiscreteBayesNet(currentModel.getName());

            // Añaidmos los nodos no-filtrados
            currentModel.getVariables().stream()
                    .filter(x->!variablesToFilter.contains(x))
                    .forEach(x -> filteredModel.addNode(x));

            // Añadimos los arcos de la variable oculta a los nodos no-filtrados
            for(DiscreteBeliefNode manifestNode: filteredModel.getManifestNodes())
                filteredModel.addEdge(manifestNode, filteredModel.getLatentNodes().get(0));

            currentModel = filteredModel;
            currentData = currentData.projectV3(currentModel.getManifestVariables());
        }
    }

    public static LearningResult<DiscreteBayesNet> learnLCM(DiscreteData data, double threshold, DiscreteParameterLearning parameterLearning) {

        DiscreteBayesNet initialModel = HlcmCreator.createLCM(data.getVariables(), 2, "wrapper_lcm", "clustVar", new Random());

        return learnLCM(data, threshold, parameterLearning, initialModel);
    }

    /**
     * Devuelve el modelo con las variables que no se ven afectadas por la variable condicionante (la maxima distancia entre sus
     * distribuciones condicionadas es inferor a un treshold).
     */
    public static DiscreteBayesNet returnFilteredModel(DiscreteBayesNet bn, DiscreteVariable conditioningVar, double threshold) {

        /* 1 - Obtenemos la lista de variables a filtrar */
        List<DiscreteVariable> variablesToFilter = filterVariables(bn, conditioningVar, threshold);

        /* 2 - Creamos un nuevo modelo */
        DiscreteBayesNet newBn = new DiscreteBayesNet();

        bn.getVariables().stream()
                .filter(x->!variablesToFilter.contains(x))
                .forEach(x -> newBn.addNode(x));

        return newBn;
    }

    /**
     * Devuelve una lista con las variables que no se ven afectadas por la variable condicionante (la maxima distancia entre sus
     * distribuciones condicionadas es inferor a un treshold).
     *
     * Dichas variables podrian ser posteriormente filtradas del modelo dada su baja relevancia.
     */
    public static List<DiscreteVariable> filterVariables(DiscreteBayesNet bn, DiscreteVariable conditioningVar, double threshold) {

        /* 1 - Conjunto de las variables hijas de la variable condicionante */
        List<DiscreteVariable> childrenVars = bn.getNodes().stream()
                .filter(x-> x.hasParent(bn.getNode(conditioningVar)))
                .map(x->x.getVariable())
                .collect(Collectors.toList());

        /* 2 - Iteramos por las variables hijas y calculamos su distancia de Hellinger condicionada */
        List<DiscreteVariable> variablesToFilter = new ArrayList<>();
        for(DiscreteVariable var: childrenVars){
            double hellingerDist = Hellinger.maxConditionalHellingerDistance(bn, conditioningVar, var);

            if(hellingerDist < threshold)
                variablesToFilter.add(var);
        }

        return variablesToFilter;
    }
}
