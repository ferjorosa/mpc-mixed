package voltric.io;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.stream.Stream;

/**
 * Simple wrapper that allow us to know current line's number for a better exception handling.
 */
public class ReaderWithLineCount {

    /** */
    private BufferedReader wrappedReader;

    /** */
    private long lineCount;

    /**
     *
     *
     * @param reader
     */
    public ReaderWithLineCount(BufferedReader reader){
        this.wrappedReader = reader;
        this.lineCount = 0;
    }

    /**
     *
     * @return
     */
    public long getLineCount(){
        return this.lineCount;
    }

    /**
     *
     * @return
     */
    public Stream<String> lines(){
        return this.wrappedReader.lines();
    }

    /**
     *
     * @return
     * @throws IOException
     */
    public String readLine() throws IOException{
        this.lineCount++; // Updates the line count
        return this.wrappedReader.readLine();
    }

    /**
     *
     * @throws IOException
     */
    public void close() throws IOException{
        this.wrappedReader.close();
    }

    /**
     *
     * @param n
     * @return
     * @throws IOException
     */
    public long skip(long n) throws IOException{
        this.lineCount = this.lineCount + n; // Updates the line count
        return this.wrappedReader.skip(n);
    }
}
