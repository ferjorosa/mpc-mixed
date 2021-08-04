package voltric.util.empiricaldist;

import voltric.data.DiscreteData;
import voltric.data.DiscreteDataInstance;
import voltric.model.DiscreteBayesNet;
import voltric.potential.Function;
import voltric.variables.DiscreteVariable;

import java.util.*;

/**
 * Created by equipo on 18/04/2018.
 *
 * TODO: La inferencia habria que actulizarla a la actual. De cualqueir forma, este metodo no genera la conjunta porque asume independencia entre las marginales P(x,y) = P(x) * P(y)
 * TODO: La manera de calcular MI entre una LV y una MV seria completando los datos y calculando la conjunta a partir de las frecuencias
 */
public class EmpDistComputer {

    private DiscreteData dataSet;

    private DiscreteBayesNet bayesNet;

    private Map<DiscreteVariable, Map<DiscreteDataInstance, Function>> latentPosts;

    public EmpDistComputer(DiscreteData dataSet, DiscreteBayesNet bayesNet){
        this.dataSet = dataSet;
        this.bayesNet = bayesNet;
        this.latentPosts = new HashMap<>();

        for(DiscreteVariable latentVar: bayesNet.getLatentVariables()){
            latentPosts.put(latentVar, new HashMap<>());
        }

        //
        createLatentPosts(this.dataSet, this.bayesNet);
    }

    public Function computeEmpDist(DiscreteVariable vFirst, DiscreteVariable vSecond){
        DiscreteVariable[] manifestVariables = bayesNet.getManifestVariables().toArray(new DiscreteVariable[bayesNet.getManifestVariables().size()]);

        List<DiscreteVariable> variablePairList = new ArrayList<>();
        variablePairList.add(vFirst);
        variablePairList.add(vSecond);
        Function empDist = Function.createFunction(variablePairList);

        for (DiscreteDataInstance dataCase : dataSet.getInstances()) {
            int[] states = dataCase.getNumericValues();

            // If there is missing data, continue;
            // TODO: Revisar
            /*if ((viIdx != -1 && states[viIdx] == -1)
                    || (vjIdx != -1 && states[vjIdx] == -1)) {
                continue;
            }*/

            // TODO: ASI NO SE GENERA UNA CONJUNTA LOL
            // P(vFirst, vSecond|d) = P(vFirst|d) * P(vSecond|d)
            Function freq;

            if(vFirst.isManifestVariable())
                freq = Function.createIndicatorFunction(vFirst, states[Arrays.binarySearch(manifestVariables, vFirst)]);
            else // vFirst is a latent variable
                freq = this.latentPosts.get(vFirst).get(dataCase);

            if(vSecond.isManifestVariable())
                freq = freq.times(Function.createIndicatorFunction(vSecond, states[Arrays.binarySearch(manifestVariables, vSecond)]));
            else // vSecond is a latent variable
                freq = freq.times(this.latentPosts.get(vSecond).get(dataCase));

            freq = freq.times(dataSet.getWeight(dataCase));
            empDist.plus(freq);
        }

        empDist.normalize();

        return empDist;
    }

    //TODO: He
    private void createLatentPosts(DiscreteData dataSet, DiscreteBayesNet bayesNet){
/*
        Map<DiscreteVariable, Map<DiscreteDataInstance, Function>> latentPosts = new HashMap<>();

        CliqueTreePropagation ctp = new CliqueTreePropagation(bayesNet);

        //Map<DiscreteVariable, Integer> varIdx = dataSet.createVariableToIndexMap();

        List<DiscreteVariable> manifestVars = bayesNet.getManifestVariables();
        int nManifestVars = manifestVars.size();

        for(DiscreteDataInstance dataCase : dataSet.getInstances()){
            // projected data instance states
            int[] manifestStates = dataCase.project(manifestVars).getNumericValues();

            // set evidence and propagate
            ctp.setEvidence(manifestVars, manifestStates);
            ctp.propagate();

            // for each of the chosen latent variables
            for(DiscreteVariable latentVar : bayesNet.getLatentVariables()){
                // compute P(Y|d)
                Function post = ctp.computeBelief(latentVar);
                Map<DiscreteDataInstance, Function> localLatentPosts = this.latentPosts.get(latentVar);
                localLatentPosts.put(dataCase, post);
            }
        }
*/
    }
}
