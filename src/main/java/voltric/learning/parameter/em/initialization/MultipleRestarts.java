package voltric.learning.parameter.em.initialization;

/**
 * TODO: Explain
 */
public class MultipleRestarts extends EmInitialization {

    public MultipleRestarts() {
        super(32, 64);
    }

    public MultipleRestarts(int nIterations, int nCandidates){
        super(nIterations, nCandidates);
    }
}
