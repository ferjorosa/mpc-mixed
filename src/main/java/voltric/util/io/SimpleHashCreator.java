package voltric.util.io;

import java.util.Collection;

/**
 * Created by equipo on 27/10/2017.
 */
public class SimpleHashCreator {

    public static <T> long createHash(Collection<T> collection){
        long sum = 0;

        for(T object: collection)
            sum += object.toString().hashCode();

        return sum;
    }

    public static <T> long createHash(T object){
        return object.toString().hashCode();
    }
}
