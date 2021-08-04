package voltric.learning.parameter.em.initialization;

public class PyramidInitialization extends EmInitialization {

    public PyramidInitialization() {
        super(16, 64);
    }

    public PyramidInitialization(int nIterations, int nCandidates){
        super(nIterations, nCandidates);
    }
}
