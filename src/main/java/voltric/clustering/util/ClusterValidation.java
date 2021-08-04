package voltric.clustering.util;

import org.apache.commons.lang3.NotImplementedException;
import voltric.data.DiscreteData;
import voltric.data.DiscreteDataInstance;
import voltric.util.Tuple;

import java.util.List;

/**
 * Created by equipo on 18/12/2017.
 */
public class ClusterValidation {

    // Basicamente hacemos el Brier score entre la distribución actual y la uniforme, de tal forma que cuanto mas lejos estemos mejor.
    // El valor máximo del BS vendria dado por sqrt((1-0.33)^2 + (0-0.33)^2 + (0-0.33)^2) con cardinalidad 3 por ejemplo
    public static double calculateNormalizedUniformBrierScore(List<Tuple<DiscreteDataInstance, double[]>> clusteringAssignments, DiscreteData data, int cardinality) {
        double[] maxAccuracyVector = new double[cardinality];
        for(int i =0; i < cardinality; i++)
            if(i == 0)
                maxAccuracyVector[i] = 1;
            else
                maxAccuracyVector[i] = 0;

        double[] uniformDistVector = new double[cardinality];
        for(int i =0; i < cardinality; i++)
            uniformDistVector[i] = 1 / cardinality;

        int n = clusteringAssignments.stream().mapToInt(x-> data.getWeight(x.getFirst())).sum();
        double maxBrierScore = calculateSingleBrierScore(maxAccuracyVector, uniformDistVector);
        double brierScore = 0;

        for(Tuple<DiscreteDataInstance, double[]> assignment: clusteringAssignments)
            brierScore += data.getWeight(assignment.getFirst()) * calculateSingleBrierScore(assignment.getSecond(), uniformDistVector);

        return brierScore / (n * maxBrierScore);
    }

    public static List<Double> calculateNormalizedBrierScores(List<Tuple<DiscreteDataInstance, double[]>> clusteringAssignments, DiscreteData data, List<Integer> cardinalities) {
        throw new NotImplementedException("Nope");
    }

    private static double calculateSingleBrierScore(double[] forecast, double[] original) {
        double brierScore = 0;

        for(int i = 0; i < forecast.length; i++)
            brierScore += Math.pow(forecast[i] - original[i], 2);

        return brierScore;
    }
}
