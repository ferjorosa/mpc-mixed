package voltric.clustering.multidimensional.old.attribute.stopcondition;

import org.apache.commons.lang3.NotImplementedException;

;

/**
 * Created by equipo on 20/04/2017.
 */
public class UnidimensionalityTest implements StopCondition{

    private double udThreshold;

    public UnidimensionalityTest(double udThreshold){
        this.udThreshold = udThreshold;
    }

    public boolean isTrue(){
        return true;
    }

    public boolean isFalse(){
        return false;
    }

    public boolean test(){
        throw new NotImplementedException("Nope");
    }
}
