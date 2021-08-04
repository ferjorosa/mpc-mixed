package voltric.learning.parameter.em;

import voltric.data.DiscreteData;
import voltric.data.DiscreteDataInstance;
import voltric.inference.LocalCliqueTreePropagation;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.config.EmConfig;
import voltric.learning.score.LearningScore;
import voltric.learning.score.ScoreType;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.potential.Function;
import voltric.variables.DiscreteVariable;

import java.util.*;

/*
    Esta version del LocalEM la vamos a probar con LocalCliqueTreePropagation.
    Podriamos hacer otra version con TrueLocalCliqueTreePropagation...

    Nota: localVariables viene definido de arriba para que sea el metodo utilizable no solo con arboles, cada algoritmo
    puede escoger el conjunto de variables que mas le interese.
*/
public class LocalEM {

    private Set<DiscreteVariable> mutableVars;

    private Random random;

    private int nSteps = 0;

    private double threshold;

    private int nMaxSteps;

    private ScoreType scoreType;

    private int pyramidCandidates;

    private int pyramidIterations;

    public LocalEM(Set<DiscreteVariable> mutableVars, EmConfig config) {
        this.mutableVars = mutableVars;
        this.random = new Random(config.getSeed());
        this.threshold = config.getThreshold();
        this.nMaxSteps = config.getnMaxSteps();
        this.scoreType = ScoreType.BIC;
        this.pyramidCandidates = config.getInitCandidates();
        this.pyramidIterations = config.getInitIterations();
    }

    public LearningResult<DiscreteBayesNet> learnModel(DiscreteBayesNet bayesNet, DiscreteData dataSet) {

        if(!dataSet.getVariables().containsAll(bayesNet.getManifestVariables()))
            throw new IllegalArgumentException("The Data set must contain all the manifest variables present in the Bayes net");

        /* resets the number of EM steps */
        this.nSteps = 0;

        /* selects a good starting point */
        LocalCliqueTreePropagation ctp = emStart(bayesNet, dataSet);

        /* Iteration[0] to compare score with iteration[1] */
        double previousScore = emStep(ctp, dataSet);
        this.nSteps++;

        /* runs EM steps until convergence */
        double score;
        do {
            score = emStep(ctp, dataSet);
            this.nSteps++;
        } while (score - previousScore > this.threshold
                && this.nSteps < this.nMaxSteps);

        return new LearningResult<>(ctp.getBayesNet(), score, this.scoreType);
    }

    public double emStep(LocalCliqueTreePropagation localCtp, DiscreteData dataSet) {

        DiscreteBayesNet bayesNet = localCtp.getBayesNet();
        HashMap<DiscreteVariable, Function> suffStats = new HashMap<>();
        Map<DiscreteVariable, Integer> evidenceValues = new HashMap<>();
        double loglikelihood = 0.0;

        for (DiscreteDataInstance dataCase : dataSet.getInstances()) {
            double weight = dataSet.getWeight(dataCase);

            /* E step: Sets the evidence using all the manifest variables in the model and then propagates it. */
            evidenceValues.clear();
            for(DiscreteVariable var: localCtp.getBayesNet().getManifestVariables())
                evidenceValues.put(var, dataCase.getNumericValue(var));
            localCtp.setEvidence(evidenceValues);
            double likelihood = localCtp.propagate();

            /* M step: Updates sufficient statistics for each of the local variables' nodes. */
            for (DiscreteVariable var : this.mutableVars) {
                Function fracWeight = localCtp.computeFamilyBelief(var);
                fracWeight.multiply(weight);
                if (suffStats.containsKey(var)) {
                    suffStats.get(var).plus(fracWeight);
                } else
                    suffStats.put(var, fracWeight);
            }

            loglikelihood += Math.log(likelihood) * weight;
        }

        /* Update BN parameters */
        for (DiscreteVariable var : this.mutableVars) {
            Function cpt = suffStats.get(var);
            cpt.normalize(var);
            bayesNet.getNode(var).setCpt(cpt);
        }

        return LearningScore.calculateScore(dataSet, bayesNet, loglikelihood, this.scoreType);
    }


    private LocalCliqueTreePropagation emStart(DiscreteBayesNet bayesNet, DiscreteData data) {
        return pyramidInitialization(bayesNet, data);
    }

    // Inicializacion piramidal. Hago lo que poon dice que hace pero que luego en su codigo no hace
    private LocalCliqueTreePropagation pyramidInitialization(DiscreteBayesNet bayesNet, DiscreteData data) {

        /* Generate initial points by randomly parameterizing BNs and assigning each one to a CTP */
        LocalCliqueTreePropagation[] ctps = new LocalCliqueTreePropagation[pyramidCandidates];
        double[] lastStepScore = new double[pyramidCandidates];
        double[] currentScore = new double[pyramidCandidates];
        for(int i = 0; i < pyramidCandidates; i++) {
            lastStepScore[i] = -Double.MAX_VALUE;
            currentScore[i] = -Double.MAX_VALUE;
        }

        /* Consider the argument BN as one of the initial points for the initialization */
        ctps[0] = new LocalCliqueTreePropagation(bayesNet.clone(), this.mutableVars);

        for (int i = 1; i < pyramidCandidates; i++) {
            DiscreteBayesNet bayesNetCopy = bayesNet.clone();

            /* Randomly parametrize BN nodes whose variables are in "mutableVars" */
            ArrayList<DiscreteBeliefNode> mutableNodesCopy = new ArrayList<>();
            for (DiscreteVariable var : this.mutableVars)
                mutableNodesCopy.add(bayesNetCopy.getNode(var));
            bayesNetCopy.randomlyParameterize(this.random, mutableNodesCopy);

            ctps[i] = new LocalCliqueTreePropagation(bayesNetCopy, this.mutableVars);
        }

        /*
         * Run EM on each candidate and retain n/2 of the candidates that led to largest values of score.
         * Then we perform two EM steps and retain n/4 candidates.  We continue this procedure, doubling the number
         * of EM steps and halving the number of candidates at each iteration until only one remains.
         */
        // TODO: Note: Poon hace una tirada de EMs previo a eliminar candidatos. Yo voy a probar a directamente aplicarlo
        // tal y como dice la teoria, y si funciona mal, lo hago como el
        int nCandidates = this.pyramidCandidates;
        int nStepsPerRound = 1;

        while (nCandidates > 1)
        {
            /* Run EM on candidates until convergence or maximum number of steps */
            for (int i = 0; i < nCandidates; i++) {
                int candidateSteps = 1;
                boolean candidateConvergence = false;
                while(!candidateConvergence && candidateSteps <= nStepsPerRound) {
                    lastStepScore[i] = currentScore[i];
                    currentScore[i] = emStep(ctps[i], data);
                    if (currentScore[i] - lastStepScore[i] <= threshold)
                        candidateConvergence = true;
                    candidateSteps++;
                }
            }

            /* Sort candidates in descending order */
            for (int i = 0; i < nCandidates - 1; i++) {
                for (int j = i + 1; j < nCandidates; j++) {
                    if (currentScore[i] < currentScore[j]) {
                        LocalCliqueTreePropagation tempCtp = ctps[i];
                        ctps[i] = ctps[j];
                        ctps[j] = tempCtp;
                    }
                }
            }

            /* Retain top half candidates and double the number of maximum EM steps */
            nCandidates /= 2;
            nStepsPerRound = Math.min(nStepsPerRound * 2, this.pyramidIterations);
        }

        /* Return the best starting point */
        return ctps[0];
    }
}
