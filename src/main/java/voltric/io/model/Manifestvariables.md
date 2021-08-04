Como BIF es un formato muy limitado, no podemos especificar que variables son ocultas y cuales
no, asi que es necesario utilizar una propiedad para definir si es latente o no.

Ejemplo

    variable Leg {
      property temporary yes;
      property latent no;
      type discrete[2] { long, short };
    }
    
Siguiendo el consejo de Poon, podriamos hacer quizas un parser como si de un compilador se tratase, para comprobar
que tanto lexico, como sintacticamente es correcto el archivo.