package methods;

import java.util.Map;

public interface BayesianMethod {

    void setPriors(Map<String, double[]> priors);
}
