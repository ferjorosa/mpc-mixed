package experiments.latent;

import methods.LatentMixedMethod;

import java.util.Set;

public abstract class LatentMixedExperiment {

    protected Set<LatentMixedMethod> methods;

    public LatentMixedExperiment(Set<LatentMixedMethod> methods) {
        this.methods = methods;
    }
}
