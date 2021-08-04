package voltric.io.model.bif;

import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.potential.Function;
import voltric.variables.DiscreteVariable;
import voltric.variables.modelTypes.VariableType;

import java.io.FileReader;
import java.io.IOException;
import java.io.StreamTokenizer;
import java.util.ArrayList;

/**
 * Los archivos OBIF deberian acabar en .obif
 *
 * TODO: Falta la property que define si una variable es latente o manifest
 * TODO: Solo lee OBIF en formato "table"
 *
 * NOTA 26-04-2018: No lee bien las tablas, utilizar XMLBifReader en vez de el, aunque sea peor
 */
@Deprecated
public class OldBifFileReader {

    public static DiscreteBayesNet readOBif(String filePath)throws IOException {

        DiscreteBayesNet bayesNet = new DiscreteBayesNet("Imported OBIF bayesian network");

        StreamTokenizer tokenizer = new StreamTokenizer(new FileReader(filePath));

        tokenizer.resetSyntax();

        // characters that will be ignored
        tokenizer.whitespaceChars('=', '=');
        tokenizer.whitespaceChars(' ', ' ');
        tokenizer.whitespaceChars('"', '"');
        tokenizer.whitespaceChars('\t', '\t');

        // word characters
        tokenizer.wordChars('A', 'z');

        // we will parse numbers
        tokenizer.parseNumbers();

        // special characters considered in the gramma
        tokenizer.ordinaryChar(';');
        tokenizer.ordinaryChar('(');
        tokenizer.ordinaryChar(')');
        tokenizer.ordinaryChar('{');
        tokenizer.ordinaryChar('}');
        tokenizer.ordinaryChar('[');
        tokenizer.ordinaryChar(']');

        // does NOT treat eol as a token
        tokenizer.eolIsSignificant(false);

        // ignores c++ comments
        tokenizer.slashSlashComments(true);

        // starts parsing
        int value;

        // reads until the end of the stream (file)
        do {
            value = tokenizer.nextToken();

            if (value == StreamTokenizer.TT_WORD) {
                // start of a new block here
                String word = tokenizer.sval;

                if (word.equals("network")) {
                    // If the first token is network, the second one should be the name of the network
                    tokenizer.nextToken();
                    bayesNet = new DiscreteBayesNet(tokenizer.sval);
                } else if (word.equals("variable")) {
                    // No name has been established for the BN, so a default one is assigned.
                    //bayesNet = new DiscreteBayesNet("Imported BIF bayesian network");

                    // parses variable. get name of variable first
                    tokenizer.nextToken();
                    String name = tokenizer.sval;

                    // looks for '['
                    do {
                        value = tokenizer.nextToken();
                    } while (value != '[');

                    // gets integer as cardinality
                    tokenizer.nextToken();
                    int cardinality = (int) tokenizer.nval;

                    // looks for '{'
                    do {
                        value = tokenizer.nextToken();
                    } while (value != '{');

                    // state list
                    ArrayList<String> states = new ArrayList<String>();

                    // gets states
                    do {
                        value = tokenizer.nextToken();
                        // If the state is represented by a word
                        if (value == StreamTokenizer.TT_WORD) {
                            String val = tokenizer.sval;
                            states.add(val);
                        }
                        // If the state is represented by a number
                        else if (value == StreamTokenizer.TT_NUMBER) {
                            String val = "" + (int) tokenizer.nval;
                            states.add(val);
                        }

                    } while (value != '}');

                    // tests consistency
                    if(states.size() != cardinality)
                        throw new IllegalArgumentException("The variable " + name + " has a number of states that differ from its cardinality");

                    // creates node
                    // Depending on the name of the node, a Manifest variable or a latent variable will be created
                    if(name.startsWith("MV_"))
                        bayesNet.addNode(new DiscreteVariable(name.replace("MV_", ""), states, VariableType.MANIFEST_VARIABLE));
                    else if(name.startsWith("LV_"))
                        bayesNet.addNode(new DiscreteVariable(name.replace("LV_", ""), states, VariableType.LATENT_VARIABLE));
                    else
                        throw new IllegalArgumentException("Variable " + name + "should start with MV_ or LV_");

                } else if (word.equals("probability")) {

                    // No name has been established for the BN, so a default one is assigned.
                    //bayesNet = new DiscreteBayesNet("Imported BIF bayesian network");

                    // parses CPT. skips next '('
                    tokenizer.nextToken();

                    // variables in this family
                    ArrayList<DiscreteVariable> family = new ArrayList<DiscreteVariable>();

                    // gets variable name and node
                    tokenizer.nextToken();
                    DiscreteBeliefNode node = bayesNet.getNode(tokenizer.sval.replace("MV_", "").replace("LV_", ""));
                    family.add(node.getVariable());

                    // gets parents and adds edges
                    do {
                        value = tokenizer.nextToken();
                        if (value == StreamTokenizer.TT_WORD) {
                            DiscreteBeliefNode parent = bayesNet.getNode(tokenizer.sval.replace("MV_", "").replace("LV_", ""));
                            family.add(parent.getVariable());

                            // adds edge from parent to node
                            bayesNet.addEdge(node, parent);
                        }
                    } while (value != ')');

                    // creates CPT
                    Function cpt = Function.createFunction(family);

                    // looks for '(' or words
                    do {
                        value = tokenizer.nextToken();
                    } while (value != '(' && value != StreamTokenizer.TT_WORD);

                    // checks next token: there are two formats, one with
                    // "table" and the other fills in cells one by one.
                    if (value == StreamTokenizer.TT_WORD) {
                        // we only accept "table" but not "default"
                        if(!tokenizer.sval.equals("table"))
                            throw new IllegalStateException("'default' is not accept, only 'table'");

                        // probability values
                        ArrayList<Double> values = new ArrayList<Double>();

                        // gets numerical tokens
                        do {
                            value = tokenizer.nextToken();
                            if (value == StreamTokenizer.TT_NUMBER) {
                                values.add(tokenizer.nval);
                            }
                        } while (value != ';');

                        // consistency between family and values will be tested
                        cpt.setCells(family, values);
                    } else {
                        // states array
                        ArrayList<Integer> states = new ArrayList<Integer>();
                        states.add(0);
                        int cardinality = node.getVariable().getCardinality();

                        // parses row by row
                        while (value != '}') {
                            // gets parent states
                            for (int i = 1; i < family.size(); i++) {
                                do {
                                    value = tokenizer.nextToken();
                                } while (value != StreamTokenizer.TT_WORD);
                                states.add(family.get(i)
                                        .indexOf(tokenizer.sval));
                            }

                            // fills in data
                            for (int i = 0; i < cardinality; i++) {
                                states.set(0, i);

                                do {
                                    value = tokenizer.nextToken();
                                } while (value != StreamTokenizer.TT_NUMBER);
                                cpt.setCell(family, states, tokenizer.nval);
                            }

                            // looks for next '(' or '}'
                            while (value != '(' && value != '}') {
                                value = tokenizer.nextToken();
                            }
                        }
                    }

                    // normalizes the CPT with respect to the attached variable
                    cpt.normalize(node.getVariable());

                    // sets the CPT
                    node.setCpt(cpt);
                }
            }
        } while (value != StreamTokenizer.TT_EOF);

        return  bayesNet;
    }
}
