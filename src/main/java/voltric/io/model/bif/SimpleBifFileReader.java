package voltric.io.model.bif;

import voltric.io.ReaderWithLineCount;
import voltric.io.model.ModelFileReader;
import voltric.model.DiscreteBayesNet;
import voltric.potential.Function;
import voltric.variables.DiscreteVariable;
import voltric.variables.modelTypes.VariableType;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * Esta version esta basada en el BIF version 0.15, pero no admite properties, lo cual suele utilizarse  mas
 * que nada para posicionamiento de la red. Lo cual no tiene sentido en esta libreria y añadiria mas trabajo
 * en la creacion de este writer y de su correspondiente reader.
 *
 * TODO: Following the specifications of this BIF version, only DISCRETE bayesian networks are supported
 *
 * TODO: Temporal, es mejor hacer un verdadero parser en Flex como si de un compilador se tratase, pero
 * en un proyecto GitHub aparte
 *
 * TODO: Hacer un metodo que elimine todas las lineas que comiencen con "property", a menos que sea property latent
 *
 * TODO: Tambien se pueden definir gramaticas especificas para cada bloque, y una gramatica general para las propiedades
 */
@Deprecated
public class SimpleBifFileReader implements ModelFileReader {

    // Podriamos evitar este parametro y que lo hiciera automatico, pero por ahora asi es mas facil
    private boolean tableFormat;

    public SimpleBifFileReader(){
        this.tableFormat = false;
    }

    public SimpleBifFileReader(boolean tableFormat){
        this.tableFormat = tableFormat;
    }

    public DiscreteBayesNet read(InputStream inputStream) throws IOException{
        // Creates a reader with incorporated line count, for a better exception handling
        // This reader wraps a BufferedReader and offers its most important methods
        ReaderWithLineCount reader = new ReaderWithLineCount(new BufferedReader(new InputStreamReader(inputStream)));
        String line;

        // The Bayes net is initialized, but empty
        DiscreteBayesNet bayesNet = new DiscreteBayesNet("Imported BIF bayesian network");

        while ((line = reader.readLine()) != null) {

            // If line is a network description line, se lee el bloque
            if(isNetworkLine(line)){
                readNetworkBlock(reader, line, bayesNet);
            }
            // Si la linea es de tipo variable, se lee el bloque
            else if(isVariableLine(line)){
                readVariableBlock(reader, line, bayesNet);
            }
            // Si la linea es de tipo probabilidad, se lee el bloque y si la variable de la que habla no existe en
            // la BN, se lnza excepcion (obligando a mantener el orden)
            else if(isProbabilityLine(line)){
                readProbabilityBlock(reader, line, bayesNet);
            }
        }
        reader.close();
        return bayesNet;
    }

    private boolean isNetworkLine(String line){
        return line.startsWith("network");
    }

    private boolean isVariableLine(String line){
        return line.startsWith("variable");
    }

    private boolean isProbabilityLine(String line){
        return line.startsWith("probability");
    }

    private boolean isEndOfBlock(String line){
        return line.startsWith("}");
    }

    private void readNetworkBlock(ReaderWithLineCount reader, String line, DiscreteBayesNet bayesNet) throws IOException{
        String[] splits = line.split(" ");

        if(splits.length != 3 || !splits[2].equals("{"))
            throw new BifParsingException("Line " + reader.getLineCount() + ": "
                    +"The network's name should have no spaces and the block should start with '{' (after the name)");

        // Sets the BN's name
        //TODO: bayesNet.setName(splits[1]);

        // TODO: No network properties allowed
        // Therefore the next line should be the end of block
        if(!isEndOfBlock(reader.readLine()))
            throw new BifParsingException("Line " + reader.getLineCount() + ": "
                    +"The network's block should not have contents and should be ended with a '}'");
    }

    private void readVariableBlock(ReaderWithLineCount reader, String line, DiscreteBayesNet bayesNet) throws IOException{
        String[] splits = line.split(" ");

        if(splits.length != 3 || !splits[2].equals("{"))
            throw new BifParsingException("Line " + reader.getLineCount() + ": "
                    +"The variable's name should have no spaces and the block should start with '{' (after the name)");

        String variableName = splits[1];

        String variableDefinitionLine = reader.readLine();

        if(isEndOfBlock(variableDefinitionLine))
            throw new BifParsingException("Line " + reader.getLineCount() + ": "
                    +"The variable block needs to specify its type, number of states and state names");

        String[] variableDefinitionParts = variableDefinitionLine.split("\\{");

        if(variableDefinitionParts.length != 2)
            throw new BifParsingException("Line " + reader.getLineCount() + ": "
                    +"Invalid variable definition");

        // 1 - The first part contains the type of the variable and the number of states if discrete
        String[] variableTypeParts = variableDefinitionParts[0].split(" ");
        if(!variableTypeParts[0].equals("type"))
            throw new BifParsingException("Line " + reader.getLineCount() + ": "
                    +"variable defintion has to start with 'type'");

        String[] secondVariableTypePartSplitted = variableTypeParts[1].split("\\[");

        if(secondVariableTypePartSplitted[0].equals("discrete"))
            throw new BifParsingException("Line " + reader.getLineCount() + ": "
                    +"Only 'discrete' variables are allowed");

        // This is the number in brackets, that corresponds to the discrete variable's number of states
        int numberOfStates = Integer.parseInt(secondVariableTypePartSplitted[1].split("\\]")[1]);

        // 2 - The second part contains the state names if the variable is discrete
        String secondVariablePart = variableTypeParts[1].trim();
        secondVariablePart = secondVariablePart.replace("}","");
        secondVariablePart = secondVariablePart.replace(";", "");

        // Now all the state names are separated with a ','
        String[] stateNames = secondVariablePart.split(",");

        if(numberOfStates != stateNames.length)
            throw new BifParsingException("Line " + reader.getLineCount() + ": "
                +"The number of states in brackets should coincide with the number of state names");

        // Read the property stating if the variable is latent or manifest
        String propertyLine = reader.readLine();

        String[] propertyLineParts = propertyLine.split(" ");
        if(     propertyLineParts.length != 3 ||
                !propertyLineParts[0].equals("property") ||
                !propertyLineParts[1].equals("latent") ||
                (!propertyLineParts[2].equals("yes") && !propertyLineParts[2].equals("no")))
            throw new BifParsingException("Line " + reader.getLineCount() + ": "
                    +"There should be a 'property' line stating if the variable is manifest/latent");

        if(propertyLineParts[2].equals("yes"))
            bayesNet.addNode(new DiscreteVariable(variableName, Arrays.asList(stateNames), VariableType.LATENT_VARIABLE));
        else
            bayesNet.addNode(new DiscreteVariable(variableName, Arrays.asList(stateNames), VariableType.MANIFEST_VARIABLE));

        // Therefore the next line should be the end of block
        if(!isEndOfBlock(reader.readLine()))
            throw new BifParsingException("Line " + reader.getLineCount() + ": "
                    +"The variable's block should be ended with a '}'");
    }

    private void readProbabilityBlock(ReaderWithLineCount reader, String line, DiscreteBayesNet bayesNet) throws IOException{
        String noSpacesLine = line.trim();

        String cptDefinition = noSpacesLine.split("\\(")[1]; //cptDefinition[0] == "probability"
        cptDefinition = cptDefinition.replace(")", ""); // The second parenthesis is eliminated
        cptDefinition = cptDefinition.replace("{", ""); // The block beginning is eliminated
        String[] cptVariables = cptDefinition.split("\\|");

        if(cptVariables.length == 1) // Variable without parents
            createOneDimensionalCpt(cptVariables[0], reader, bayesNet);
        else if(cptVariables.length > 1) // Variable with 1 or more parents
            createMultidimensionalCpt(cptVariables, reader, bayesNet);
        else
            throw new BifParsingException("Line " + reader.getLineCount() + ": "
                    +"Incorrect probability definition");
    }

    private void createOneDimensionalCpt(String variable, ReaderWithLineCount reader, DiscreteBayesNet bayesNet) throws IOException{
        if(bayesNet.getNode(variable) == null)
            throw new BifParsingException("Line " + reader.getLineCount() + ": "
                    +"Variable '"+variable+"' needs to be previously defined");

        // First we add the CPT variables
        List<DiscreteVariable> cptVariables = new ArrayList<DiscreteVariable>();
        cptVariables.add(bayesNet.getNode(variable).getVariable());
        Function cpt = Function.createFunction(cptVariables);

        // Then we set the CPT values (cells)
        String cptValuesLine = reader.readLine();
        String[] valueParts = cptValuesLine.split(" ");
        if(!valueParts[0].equals("table"))
            throw new BifParsingException("Line " + reader.getLineCount() + ": "
                    +"Line has to start with 'table'");

    }

    private void createMultidimensionalCpt(String[] variables, ReaderWithLineCount reader, DiscreteBayesNet bayesNet) throws IOException{

    }
}
