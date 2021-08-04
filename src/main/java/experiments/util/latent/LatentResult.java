package experiments.util.latent;

import java.util.Map;

public class LatentResult {

    private double average_learning_time;
    private double average_test_LL;
    private double average_train_BIC;
    private Map<String, LatentFoldResult> folds;

    public double getAverage_learning_time() {
        return average_learning_time;
    }

    public void setAverage_learning_time(double average_learning_time) {
        this.average_learning_time = average_learning_time;
    }

    public double getAverage_test_LL() {
        return average_test_LL;
    }

    public void setAverage_test_LL(double average_test_LL) {
        this.average_test_LL = average_test_LL;
    }

    public double getAverage_train_BIC() {
        return average_train_BIC;
    }

    public void setAverage_train_BIC(double average_train_BIC) {
        this.average_train_BIC = average_train_BIC;
    }

    public Map<String, LatentFoldResult> getFolds() {
        return folds;
    }

    public void setFolds(Map<String, LatentFoldResult> folds) {
        this.folds = folds;
    }
}
