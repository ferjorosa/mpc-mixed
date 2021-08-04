package voltric.io.model.bif;

/**
 * Created by fernando on 11/04/17.
 */
@Deprecated
class BifParsingException extends RuntimeException {

    /**
     * Default constructor
     */
    BifParsingException() { super(); }

    /**
     * Constructs an {@code BifParsingException} with a specific message.
     *
     * @param message the argument message
     */
    BifParsingException(String message) { super(message); }
}
