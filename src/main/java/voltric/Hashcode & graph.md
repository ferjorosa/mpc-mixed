El hecho de utilizar una librería propia de grafos da muchisimos problemas a nivel de hashcode
y equals(), es un verdadero lio la comparativa y el hehco de utilizar "uniqueID" no mejora
las cosas

Hay que tener mucho cuidado al introducir elementos mutables en el HashCode, signifca que dichos
objetos no deberian formar parte de las keys de un hashmap si se modifican.