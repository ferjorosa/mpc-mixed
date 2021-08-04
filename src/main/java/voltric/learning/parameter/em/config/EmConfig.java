package voltric.learning.parameter.em.config;

import voltric.learning.parameter.em.initialization.EmInitialization;
import voltric.learning.parameter.em.initialization.PyramidInitialization;
import voltric.learning.score.ScoreType;

import java.util.HashSet;

/**
 * Created by fernando on 3/04/17.
 */
public class EmConfig {

    /** The threshold to control the algorithm's convergence */
    protected double threshold;

    /** The maximum number of EM steps to control its convergence */
    protected int nMaxSteps;

    /** The escape method is used to choose a good starting point for the EM algorithm */
    protected EmInitialization initializationMethod;

    /** The flag indicates whether we reuse the parameters of the input BN as a candidate starting point. */
    protected boolean reuse = true;

    protected ScoreType scoreType;

    /** The collection of nodes that shouldnt be updated by the EM algorithm */
    protected HashSet<String> dontUpdateNodes;

    protected long seed;

    public EmConfig(long seed){
        this.threshold = 0.01;
        this.nMaxSteps = 500;
        this.initializationMethod = new PyramidInitialization();
        this.reuse = false;
        this.scoreType = ScoreType.BIC;
        this.dontUpdateNodes = new HashSet<>();
        this.seed = seed;
    }

    public EmConfig(long seed,
                    double threshold,
                    int nMaxSteps,
                    EmInitialization initializationMethod,
                    boolean reuse,
                    ScoreType scoreType,
                    HashSet<String> dontUpdateNodes){

        this.threshold = threshold;
        this.nMaxSteps = nMaxSteps;
        this.initializationMethod = initializationMethod;
        this.reuse = reuse;
        this.scoreType = scoreType;
        this.dontUpdateNodes = dontUpdateNodes;
        this.seed = seed;
    }

    public double getThreshold() {
        return threshold;
    }

    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }

    public int getnMaxSteps() {
        return nMaxSteps;
    }

    public void setnMaxSteps(int nMaxSteps) {
        this.nMaxSteps = nMaxSteps;
    }

    public EmInitialization getInitializationMethod() {
        return initializationMethod;
    }

    public void setInitializationMethod(EmInitialization initializationMethod) {
        this.initializationMethod = initializationMethod;
    }

    public boolean isReuse() {
        return reuse;
    }

    public void setReuse(boolean reuse) {
        this.reuse = reuse;
    }

    public HashSet<String> getDontUpdateNodes() {
        return dontUpdateNodes;
    }

    public void setDontUpdateNodes(HashSet<String> dontUpdateNodes) {
        this.dontUpdateNodes = dontUpdateNodes;
    }

    public int getInitIterations() { return initializationMethod.getIterations(); }

    public int getInitCandidates() { return initializationMethod.getCandidates(); }

    public void setInitIterations(int nIterations) {
        this.initializationMethod.setnIterations(nIterations);
    }

    public void setInitCandidates(int nCandidates) {
        this.initializationMethod.setnCandidates(nCandidates);
    }

    public ScoreType getScoreType() { return scoreType; }

    public void setScoreType(ScoreType scoreType) { this.scoreType = scoreType; }

    public long getSeed() { return seed; }

    public void setSeed(long seed) { this.seed = seed; }
}
