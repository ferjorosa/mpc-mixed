package voltric.graph.exception;

/**
 * Este tipo de excepcion se crea para aquellos casos como en el Hill climbing donde la excepción es algo "normal", y por
 * tanto se puede recoger tranquilamente
 */
public class IllegalEdgeException extends RuntimeException {

    public IllegalEdgeException() {
        super();
    }

    public IllegalEdgeException(String s) {
        super(s);
    }

    public IllegalEdgeException(String s, Throwable throwable) {
        super(s, throwable);
    }

    public IllegalEdgeException(Throwable throwable) {
        super(throwable);
    }
}

