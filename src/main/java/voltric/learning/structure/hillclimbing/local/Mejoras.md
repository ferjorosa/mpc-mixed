Dado que isEdgeAllowed puede ser muy costoso, quizas es mejor aprender todos los arcos y despues, una vez descartados
aquellos que no mejoran el score, probar si el arco se encuentra permitido

EXTRA: Dado que ahora devolvemos el conjunto de operaciones, quizas tiene mas sentido calcular todos y despues:
- ~~Ordenar por score las operaciones.~~ **(Ya se hace)**

- Mirar si no forma un ciclo **(Se hace por ahora dentro de los operadores)**

- Entonces mirar si la estructura sigue las especificaciones (Forest, PolyForest, DAG)
    - TODO: Ambas operaciones de mirar si no forma un ciclo y demas se pueden hacer desde StructureType.allows(copyDag)