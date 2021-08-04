## IO/Data
Estaria bien que al cargar datos ARFF se comprobase que NO existen instancias sin atributo asociado, para que no pase
que durante el aprendizaje se le asigne -1 y nos comamos una excepcion que no viene a cuento.