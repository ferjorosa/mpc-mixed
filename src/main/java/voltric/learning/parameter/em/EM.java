package voltric.learning.parameter.em;

import org.apache.commons.lang3.NotImplementedException;
import voltric.data.DiscreteData;
import voltric.data.DiscreteDataInstance;
import voltric.inference.CliqueTreePropagation;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.config.EmConfig;
import voltric.learning.score.LearningScore;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.model.HLCM;
import voltric.potential.Function;
import voltric.variables.DiscreteVariable;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

;

/**
 * Created by fernando on 4/04/17.
 */
// TODO: Crear una nueva clase EM y rehacer la inicializacion para que sea mas eficiente (como en LocalEM)
public class EM extends AbstractSequentialEM {

    private Random random;

    public EM(EmConfig config) {
        super(config);
        this.random = new Random(config.getSeed());
    }

    /** {@inheritDoc} */
    //TODO: La red bayesNet solo se utiliza de base
    @Override
    public LearningResult<DiscreteBayesNet> learnModel(DiscreteBayesNet bayesNet, DiscreteData dataSet) {

        DiscreteData projectedData = dataSet.project(bayesNet.getManifestVariables());

        // resets the number of EM steps
        this.nSteps = 0;

        // selects a good starting point
        CliqueTreePropagation ctp = emStart(bayesNet, projectedData);

        double previousScore = emStep(ctp, projectedData);
        this.nSteps++;

        // runs EM steps until convergence or if the max number of steps is reached.
        // It is necessary to do at least one iteration
        double score;
        do {
            score = emStep(ctp, projectedData);
            this.nSteps++;
        } while (score - previousScore > threshold
                && this.nSteps < nMaxSteps);

        return new LearningResult<>(ctp.getBayesNet(), score, this.scoreType);
    }

    /** {@inheritDoc} */
    @Override
    public double emStep(CliqueTreePropagation ctp, DiscreteData dataSet){
        // gets the BN to be optimized
        DiscreteBayesNet bayesNet = ctp.getBayesNet();

        // sufficient statistics for each node
        HashMap<DiscreteVariable, Function> suffStats = new HashMap<DiscreteVariable, Function>();

        double loglikelihood = 0.0;
        Map<DiscreteVariable, Integer> evidenceValues = new HashMap<>();

        for (DiscreteDataInstance dataInstance : dataSet.getInstances()) {
            double weight = dataSet.getWeight(dataInstance);

            // sets evidences
            evidenceValues.clear();
            for(DiscreteVariable var: ctp.getBayesNet().getManifestVariables())
                evidenceValues.put(var, dataInstance.getNumericValue(var));
            ctp.setEvidence(evidenceValues);

            // propagates
            double likelihoodDataCase = ctp.propagate();

            // updates sufficient statistics for each node
            for (DiscreteVariable var : bayesNet.getVariables()) {

                if(this.dontUpdateNodes != null && this.dontUpdateNodes.contains(var.getName()))
                    continue;

                Function fracWeight = ctp.computeFamilyBelief(var);

                fracWeight.multiply(weight);

                if (suffStats.containsKey(var)) {
                    suffStats.get(var).plus(fracWeight);
                } else {
                    suffStats.put(var, fracWeight);
                }
            }

            loglikelihood += Math.log(likelihoodDataCase) * weight;

        }

        // updates parameters
        for (DiscreteBeliefNode node : bayesNet.getNodes()) {

            if(this.dontUpdateNodes != null && this.dontUpdateNodes.contains(node.getVariable().getName()))
                continue;

            Function cpt = suffStats.get(node.getVariable());
            cpt.normalize(node.getVariable());
            node.setCpt(cpt);
        }

        return LearningScore.calculateScore(dataSet, bayesNet, loglikelihood, this.scoreType);
    }

    protected CliqueTreePropagation pyramidInitialization(DiscreteBayesNet bayesNet, DiscreteData data) {
        /* Generate initial points by randomly parameterizing BNs and assigning each one to a CTP */
        CliqueTreePropagation[] ctps = new CliqueTreePropagation[nInitCandidates];
        double[] lastStepScore = new double[nInitCandidates];
        double[] currentScore = new double[nInitCandidates];
        for(int i = 0; i < nInitCandidates; i++) {
            lastStepScore[i] = -Double.MAX_VALUE;
            currentScore[i] = -Double.MAX_VALUE;
        }

        /* Consider the argument BN as one of the initial points for the initialization */
        ctps[0] = new CliqueTreePropagation(bayesNet.clone());

        for (int i = 1; i < nInitCandidates; i++) {
            DiscreteBayesNet bayesNetCopy = bayesNet.clone();
            bayesNetCopy.randomlyParameterize(random);
            ctps[i] = new CliqueTreePropagation(bayesNetCopy);
        }

        /*
         * Run EM on each candidate and retain n/2 of the candidates that led to largest values of score.
         * Then we perform two EM steps and retain n/4 candidates.  We continue this procedure, doubling the number
         * of EM steps and halving the number of candidates at each iteration until only one remains.
         */
        // TODO: Note: Poon hace una tirada de EMs previo a eliminar candidatos. Yo voy a probar a directamente aplicarlo
        // tal y como dice la teoria, y si funciona mal, lo hago como el
        int nCandidates = this.nInitCandidates;
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
                        CliqueTreePropagation tempCtp = ctps[i];
                        ctps[i] = ctps[j];
                        ctps[j] = tempCtp;
                    }
                }
            }

            /* Retain top half candidates and double the number of maximum EM steps */
            nCandidates /= 2;
            nStepsPerRound = Math.min(nStepsPerRound * 2, this.nInitIterations);
        }

        /* Return the best starting point */
        return ctps[0];

    }

    /** {@inheritDoc} */
    @Override
    protected CliqueTreePropagation chickeringHeckermanInitialization(DiscreteBayesNet bayesNet, DiscreteData dataSet) {
        // generates random starting points and CTPs for them
        CliqueTreePropagation[] ctps = new CliqueTreePropagation[this.nInitCandidates];
        double[] lastStepScore = new double[this.nInitCandidates];
        double[] currentScore = new double[this.nInitCandidates];

        for (int i = 0; i < this.nInitCandidates; i++) {
            DiscreteBayesNet copy = bayesNet.clone();

            // in case we reuse the parameters of the input BN as a starting
            // point, we put it at the first place.
            if (!this.reuse || i != 0)
            {
                if(this.dontUpdateNodes == null)
                {
                    copy.randomlyParameterize(this.random);
                }else
                {
                    for(DiscreteBeliefNode node : copy.getNodes())
                    {
                        if(!this.dontUpdateNodes.contains(node.getVariable().getName()))
                        {
                            Function cpt = node.getCpt();
                            cpt.randomlyDistribute(node.getVariable(), this.random);
                            node.setCpt(cpt);
                        }
                    }
                }
            }

            if (copy instanceof HLCM) {
                ctps[i] = new CliqueTreePropagation((HLCM) copy);
            } else {
                ctps[i] = new CliqueTreePropagation(copy);
            }
        }

        // We run several steps of emStep before killing starting points for two reasons:
        // 1. the loglikelihood-related score being computed is always greater that of previous model.
        // 2. When reuse, the reused model is kind of dominant because maybe it has already EMed.
        this.nSteps += this.nInitIterations;
        for (int i = 0; i < this.nInitCandidates; i++) {
            double score = 0;
            for (int j = 0; j < nInitIterations; j++)
                score = emStep(ctps[i], dataSet);

            currentScore[i] = score;
        }

        // game starts, half ppl die in each round :-)
        int nCandidates = this.nInitCandidates;
        int nStepsPerRound = 1;

        while (nCandidates > 1 && this.nSteps < this.nMaxSteps)
        {
            // runs EM on all starting points for several steps
            for (int j = 0; j < nStepsPerRound; j++)
            {
                boolean noImprovements = true;
                for (int i = 0; i < nCandidates; i++)
                {
                    lastStepScore[i] = currentScore[i];
                    currentScore[i] = emStep(ctps[i], dataSet);

                    if(currentScore[i] - lastStepScore[i] > this.threshold || lastStepScore[i] == Double.NEGATIVE_INFINITY)
                        noImprovements = false;
                }
                this.nSteps++;

                if(noImprovements)
                    return ctps[0];

            }

            // sorts BNs in descending order with respect to the score
            for (int i = 0; i < nCandidates - 1; i++) {
                for (int j = i + 1; j < nCandidates; j++) {
                    if (currentScore[i] < currentScore[j]) {
                        CliqueTreePropagation tempCtp = ctps[i];
                        ctps[i] = ctps[j];
                        ctps[j] = tempCtp;

                    }
                }
            }

            // retains top half
            nCandidates /= 2;

            // doubles EM steps subject to maximum step constraint
            nStepsPerRound = Math.min(nStepsPerRound * 2, this.nMaxSteps - this.nSteps);
        }

        // returns the CTP for the best starting point
        return ctps[0];
    }

    /** {@inheritDoc} */
    @Override
    protected CliqueTreePropagation randomInitialization(DiscreteBayesNet bayesNet, DiscreteData dataSet){
        throw new NotImplementedException("Nope");
    }
}
