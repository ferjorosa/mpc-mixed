package voltric.learning;

import voltric.learning.score.ScoreType;
import voltric.model.AbstractBayesNet;

/**
 * Created by fernando on 7/08/17.
 */
public class LearningResult<B extends AbstractBayesNet> {

    private double scoreValue;

    private B bayesianNetwork;

    private ScoreType scoreType;

    private String name;

    public LearningResult(B bayesianNetwork, double scoreValue, ScoreType scoreType){
        this.bayesianNetwork = bayesianNetwork;
        this.scoreValue = scoreValue;
        this.scoreType = scoreType;
        this.name = "";
    }

    public LearningResult(B bayesianNetwork, double scoreValue, ScoreType scoreType, String name){
        this.bayesianNetwork = bayesianNetwork;
        this.scoreValue = scoreValue;
        this.scoreType = scoreType;
        this.name = name;
    }

    public double getScoreValue() {
        return scoreValue;
    }

    public B getBayesianNetwork() {
        return bayesianNetwork;
    }

    public ScoreType getScoreType() {
        return scoreType;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }
}
