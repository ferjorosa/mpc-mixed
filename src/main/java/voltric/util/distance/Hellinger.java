package voltric.util.distance;

import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.potential.Function;
import voltric.variables.DiscreteVariable;

import java.util.ArrayList;
import java.util.List;

/**
 * La distancia de Hellinger se encuentra fuertemente relacionada con la distancia de Bhattacharyya
 * H^{2}(P,Q) = 1 - BC(P,Q) => H(P,Q) = sqrt(1 - BC(P,Q))
 */
public class Hellinger {

    public static double distance(Function a, Function b) {
        return Math.sqrt(1 - Bhattacharyya.distance(a,b));
    }

    // Cuando son HLCMs las hijas de una LV pueden ser del tipo manifest o latent, para calcular la distancia
    public static List<double[][]> clusterDistances(DiscreteBayesNet mcm) {

        List<double[][]> bhattacharyyaDistances = Bhattacharyya.clusterDistances(mcm);
        List<double[][]> hellingerDistances = new ArrayList<>(bhattacharyyaDistances.size());

        for(double[][] bhattacharyyaMatrix: bhattacharyyaDistances){
            double[][] distanceMatrix = new double[bhattacharyyaMatrix.length][bhattacharyyaMatrix.length];

            for(int i = 0; i< bhattacharyyaMatrix.length; i++)
                for(int j = 0; j < bhattacharyyaMatrix.length; j++)
                    distanceMatrix[i][j] = 0;

            for(int i = 0; i < bhattacharyyaMatrix.length; i++)
                for(int j = 0; j < bhattacharyyaMatrix.length; j++)
                    if(i != j)
                        distanceMatrix[i][j] = Math.sqrt(1 - bhattacharyyaMatrix[i][j]);

            hellingerDistances.add(distanceMatrix);
        }
        return hellingerDistances;
    }

    public static List<Double> averageClusterDistances(DiscreteBayesNet mcm) {

        List<double[][]> hellingerMatrixes = clusterDistances(mcm);

        List<Double> partitionHellingerValues = new ArrayList<>();

        double sumHellingerValues = 0;

        for(double[][] hellingerMatrix: hellingerMatrixes){
            sumHellingerValues = 0;
            List<Double> hellingerValues = new ArrayList<>();

            // Iteramos por la upper triangular matrix sin contar la diagonal que es 0
            for (int i = 0; i < hellingerMatrix.length; i++){
                for (int j = i; j < hellingerMatrix.length; j++)
                    if (j != i)
                        hellingerValues.add(hellingerMatrix[i][j]);
            }

            for(double value: hellingerValues)
                sumHellingerValues += value;

            // Añadimos la distancia media de Hellinger para la particion en cuestion
            partitionHellingerValues.add(sumHellingerValues / hellingerValues.size());
        }

        return partitionHellingerValues;
    }

    /** Calcula la matriz de distancias de Hellinger cuando el input es un LCM */
    public static double[][] clusterDistancesLCM(DiscreteBayesNet lcm) {
        double[][] bhattacharyyaDistances = Bhattacharyya.clusterDistancesLCM(lcm);

        double[][] distanceMatrix = new double[bhattacharyyaDistances.length][bhattacharyyaDistances.length];
        for(int i = 0; i< bhattacharyyaDistances.length; i++)
            for(int j = 0; j < bhattacharyyaDistances.length; j++)
                distanceMatrix[i][j] = 0;

        for(int i = 0; i < bhattacharyyaDistances.length; i++)
            for(int j = 0; j < bhattacharyyaDistances.length; j++)
                if(i != j)
                    distanceMatrix[i][j] = Math.sqrt(1 - bhattacharyyaDistances[i][j]);

        return distanceMatrix;
    }

    public static double averageClusterDistancesLCM(DiscreteBayesNet lcm) {
        double[][] hellingerDistances = clusterDistancesLCM(lcm);

        List<Double> hellingerValues = new ArrayList<>();

        // Iteramos por la upper triangular matrix sin contar la diagonal que es 0
        for (int i = 0; i < hellingerDistances.length; i++){
            for (int j = i; j < hellingerDistances.length; j++)
                if (j != i)
                    hellingerValues.add(hellingerDistances[i][j]);
        }

        double sumHellingerValues = 0;
        for(double value: hellingerValues)
            sumHellingerValues += value;

        return sumHellingerValues / hellingerValues.size();
    }

    /**
     * Calcula las distancia de Hellinger maxima entre la distribucion marginal y la condicionada segun la conditioning var
     * para cada uno de los estados de la variable condicionante.
     *
     * Esta pensado para LCMs.
     */
    public static double maxConditionalHellingerDistance(DiscreteBayesNet bn, DiscreteVariable conditioningVar, DiscreteVariable variable) {
        if(!bn.containsVar(variable))
            throw new IllegalArgumentException("The variable must belong to the BN");

        DiscreteBeliefNode node = bn.getNode(variable);

        /* 1 - Iteramos por cada uno de los estaados de la variable raiz y proyectamos la CPT correspondiente */
        Function[] projectedCPTs = new Function[conditioningVar.getCardinality()];
        for(int i = 0; i < conditioningVar.getCardinality(); i++)
            projectedCPTs[i] = node.getCpt().project(conditioningVar, i);

        /* 2 - Calculamos la distancia entre cada par de CPTs y almacenamos la distancia maxima */
        double maxDistance = -1;
        for(int i = 0; i < projectedCPTs.length; i++)
            for(int j = i + 1; j < projectedCPTs.length; j++){
                double currentDistance = distance(projectedCPTs[i], projectedCPTs[j]);

                if(currentDistance > maxDistance)
                    maxDistance = currentDistance;
            }

        return maxDistance;
    }

    /**
     * Calcula las distancias de Hellinger entre la distribucion marginal y la condicionada segun la conditioning var
     * para cada uno de los estados de la variable condicionante.
     *
     * Esta pensado para LCMs.
     */
    public static double[][] conditionalHellingerDistances(DiscreteBayesNet bn, DiscreteVariable conditioningVar, DiscreteVariable variable) {
        if(!bn.containsVar(variable))
            throw new IllegalArgumentException("The variable must belong to the BN");

        DiscreteBeliefNode node = bn.getNode(variable);

        /* 1 - Iteramos por cada uno de los estaados de la variable raiz y proyectamos la CPT correspondiente */
        Function[] projectedCPTs = new Function[conditioningVar.getCardinality()];
        for(int i = 0; i < conditioningVar.getCardinality(); i++)
            projectedCPTs[i] = node.getCpt().project(conditioningVar, i);

        /* 2 - Calculamos la distancia entre cada par de CPTs*/
        double[][] distances = new double[projectedCPTs.length][projectedCPTs.length];
        for(int i = 0; i < projectedCPTs.length; i++)
            for(int j = i + 1; j < projectedCPTs.length; j++){
                distances[i][j] = distance(projectedCPTs[i], projectedCPTs[j]);
                distances[j][i] = distances[i][j];//symmetric
            }

        return distances;
    }
}
