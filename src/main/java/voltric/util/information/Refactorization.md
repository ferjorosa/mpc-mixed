Tras eliminar JointDistributionMI se deberia hacer publico las diferentes maneras de calcular
la MI, de tal forma que no quede duda que tipo de calculo se esta haciendo y que consecuencias tiene.

El único sitio donde he visto que quizas tiene sentido el dejar un interfaz común para ambos es a nivel de
tests estadisiticos (package voltric.util.stattest.discrete). Pero incluso ahi creo que habria
que distinguir.