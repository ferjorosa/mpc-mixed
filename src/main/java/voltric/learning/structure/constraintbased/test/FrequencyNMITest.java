package voltric.learning.structure.constraintbased.test;

import voltric.data.DiscreteData;
import voltric.util.information.mi.NMI;
import voltric.util.information.mi.normalization.NormalizationFactor;
import voltric.variables.DiscreteVariable;

import java.util.List;

/**
 * Requiere optimizacion, ya que vamos a llamar a FreqNMI, pero podriamos almacenar las probabilidades marginales para
 * reutilizarlas en varios tests.
 */

// TODO: Testear que funciona bien con un paso de condicionales vacio (caso 0)
public class FrequencyNMITest implements CITest{

    private NormalizationFactor nmiNormalizationFactor;

    public FrequencyNMITest(NormalizationFactor nmiNormalizationFactor){
        this.nmiNormalizationFactor = nmiNormalizationFactor;
    }

    public double test(DiscreteVariable a, DiscreteVariable b, List<DiscreteVariable> conditionalVars, DiscreteData data){

        // First compute the NCMI
        double ncmi = NMI.computeConditional(a,b,conditionalVars, data, this.nmiNormalizationFactor);

        // TODO: Hablar con PyC porque no entiendo bien que deberia salir aqui, hablar de las transparencias y pedir referencia articulo
        /*
        // Now, we do a hypothesis test using the chi-square distribution, given that 2NMI -> chi-square dist with parameter k = (|a| - 1) * (|b| - 1)
        double degreesOfFreedom = (a.getCardinality() - 1) * (b.getCardinality() - 1);
        ChiSquaredDistribution chi2dist = new ChiSquaredDistribution(degreesOfFreedom);

        return chi2dist.density(2 * ncmi);
        */

        return ncmi;
    }
}
