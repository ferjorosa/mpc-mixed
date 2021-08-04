package voltric.learning.parameter.em.initialization;

/**
 * This abstract class groups the s initialization strategies common code. The EM's initialization techniques allow it
 * to choose a good starting point for its subsequent steps.
 */
public abstract class EmInitialization {

    /** The number of EM steps executed for each candidate during initialization. */
    protected int nIterations;

    /** The number of random points considered during initialization. */
    protected int nCandidates;

    public EmInitialization(int nIterations, int nCandidates){
        this.nIterations = nIterations;
        this.nCandidates = nCandidates;
    }

    public int getIterations() { return nIterations; }

    public int getCandidates() { return nCandidates; }

    public void setnIterations(int nIterations) {
        this.nIterations = nIterations;
    }

    public void setnCandidates(int nCandidates) {
        this.nCandidates = nCandidates;
    }
}
