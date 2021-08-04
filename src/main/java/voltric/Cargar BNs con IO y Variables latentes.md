Debe de existir una equivalencia de orden entre las variables en la BN y en el DataSet. Si una variable
ha sido cargada la primera (ya sea porque era la primera en un archivo BIF o porque se ha creado manualmente),
deberia estar la primera en el set de Datos. 

Esto da problemas cuando completamos datos a partir de un modelo con variables latentes, ya que al completar
pasan a estar dichas columnas en ultima posicion. Sin embargo, si dicho modelo ha sido cargado de un archivo
BIF suele estar dicha variable en primer lugar. Por eso independientemente de su lugar en el archivo, debe
de ser colocada en ULTIMO LUGAR.

### Combinar BNs y orden de las LVs
Cuando combinamos varios modelos con variables latentes en uno solo, es necesario que dicho modelo mantenga
lo dicho anteriormente.

Nota: A su vez hay que tener cuidado sobre el orden mismo de dichas LVs para que al completar los datos 
coincida tambien el orden.

### Index de las LVs
Mucho cuidado al cargar variables latentes del modelo, aunque las coloquemos las ultimas, si han sido 
creadas las primeras su indice será el primero, lo cual liaria aun mas el tema de los indices en las CPTs

La clave es ver que en funciones siempre esten las ultimas, ya que su indice deberia ser el ultimo.

### Index de las MVs y aprendizaje
De la misma forma que las variables latentes tienen problemas de indices, tambien lo tienen las variables manifest.
Cuando cargamos un modelo e intentamos modificar su CPT con una nueva mediante setCPT(), al contar con indice diferente
nos dara un error, ya que sus indices no coincidiran.

Esto es importante para LocalHC, ya que en performOperation() se llama a setCpt(), aunque habra seguro mas casos.

Ironicamente el indice esta fuera del hashcode asi que uno de ellos puede funcionar mientras que el otro no, y es que el 
indice solo se utiliza para ordenar...

COMO LO HE RESULETO TEMPORALMENTE: mediante setIndex() y adjustVariableIndexes en DiscreteBayesnet

### Crear modelos latentes por codigo (model.creator)
Es necesario que cuando creemos un modelo latente como por ejemplo un LCM, dicha variable latente se 
encuentre en ultimo lugar, de tal forma que si se exporta y se lee con XMLBifReader no haya problemas
y sus CPTs se correspondan