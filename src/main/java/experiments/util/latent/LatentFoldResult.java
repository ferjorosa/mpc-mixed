package experiments.util.latent;

public class LatentFoldResult {

    private double test_LL;
    private double train_BIC;
    private long learning_time;

    public double getTest_LL() {
        return test_LL;
    }

    public void setTest_LL(double test_LL) {
        this.test_LL = test_LL;
    }

    public double getTrain_BIC() {
        return train_BIC;
    }

    public void setTrain_BIC(double train_BIC) {
        this.train_BIC = train_BIC;
    }

    public long getLearning_time() {
        return learning_time;
    }

    public void setLearning_time(long learning_time) {
        this.learning_time = learning_time;
    }
}
