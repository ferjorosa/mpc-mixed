package voltric.util.information.mi;

import voltric.data.DiscreteData;
import voltric.util.information.entropy.FrequencyCountedEntropy;
import voltric.variables.DiscreteVariable;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by equipo on 19/10/2017.
 */
public class FrequencyCountedMI {

    public static double computeConditional(DiscreteVariable x, DiscreteVariable y, DiscreteVariable condVar, DiscreteData data){
        List<DiscreteVariable> xList = new ArrayList<>();
        xList.add(x);

        List<DiscreteVariable> yList = new ArrayList<>();
        yList.add(y);

        List<DiscreteVariable> condVars = new ArrayList<>();
        condVars.add(condVar);

        return computeConditional(xList, yList, condVars, data);
    }

    public static double computeConditional(DiscreteVariable x, DiscreteVariable y, List<DiscreteVariable> condVars, DiscreteData data){
        List<DiscreteVariable> xList = new ArrayList<>();
        xList.add(x);

        List<DiscreteVariable> yList = new ArrayList<>();
        yList.add(y);
        return computeConditional(xList, yList, condVars, data);
    }

    public static double computeConditional(List<DiscreteVariable> x, List<DiscreteVariable> y, List<DiscreteVariable> condVars, DiscreteData data){

        if(!data.getVariables().containsAll(x) || !data.getVariables().containsAll(y) || !data.getVariables().containsAll(condVars))
            throw new IllegalArgumentException("The dataSet must contain all the variables X, Y & condVars");

        List<DiscreteVariable> xyz = new ArrayList<>();
        xyz.addAll(x);
        xyz.addAll(y);
        xyz.addAll(condVars);

        List<DiscreteVariable> xz = new ArrayList<>();
        xz.addAll(x);
        xz.addAll(condVars);

        List<DiscreteVariable> yz = new ArrayList<>();
        yz.addAll(y);
        yz.addAll(condVars);

        double Hxz = FrequencyCountedEntropy.compute(data.project(xz));
        double Hyz = FrequencyCountedEntropy.compute(data.project(yz));
        double Hxyz = FrequencyCountedEntropy.compute(data.project(xyz));
        double Hz = FrequencyCountedEntropy.compute(data.project(condVars));

        return Hxz + Hyz - Hxyz - Hz;
    }
}
