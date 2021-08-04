TODO: 03-09-2017

Basicamente hay que eliminar el metodo

AbstractNode<T> addNode(AbstractNode<T> node);

Este metodo nos da varios problemas porque al ser publico y utilizarse en los grafos Weighted y en las BayesNet juntamos
dos tipos de estructuras que no deberian estar relacionadas directamente. Siguiendo los pasos anteriores
de dividir las funciones de cada clase, lo que se hara es definir un "Envelope" para los nodos del grafo, dichos
envoltorios (composite pattern) no dependeran de la jerarquia de DirectedNode<T>

Solo se pueden crear nodos pasandoles como argumento el contenido, nunca pasando una referencia de otro nodo,
sigue aspectos de inmutabilidad.

Cambio: Actualmente solo se utiliza en DiscreteBayesNet, lo que me da mas razones para eliminar dicho metodo y cambiar las cosas
Ademas que da pie a nuevos tipos de estructuras como son los diagramas de influencia, etc. Y en un futuro puede
que permita la introducion de nuevos modelos o de incluir mas restricciones como Conjugate Expontential models

TODO: 24-10-2017

Hay varios problems con el diseño actual:

- Solo existe un tipo de Arco: dirigido
- Los BeliefNodes heradan de los nodes y al juntar la respresentacion de la BN con el grafo nos provoca un monton de problemas.
Por ejemplo, necesitamos de IDs unicos para que no se añadan nodos de otras redes a nuestra red, lo cual es estupido porque deberiamos
poder reutilizar nodos entre varias redes, si bien cada red bayesiana deberia tener sus propias distribuciones las cuales
se almancenarian en listas

Solucion:

- Principalmente yo abogo por trabajar con un "template" mas abstracto que permita utilizar varias librerias de grafos
y redefinir el paquete de modelo de tal forma que exista una separación de conceptos, habria que crear una rama en GitHub para ello, sino
seria un lio y se podria perder trabajo hecho.