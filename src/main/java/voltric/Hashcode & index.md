Me estoy encontrando problemas con el HashCode & index, ya que al incrementar/decrementar la cardinalidad del modelo
se genera una nueva variable latente con un nuevo indice.

Como dicho indice es posterior al inicial desvirtua seguramente los dataSets generados

Vamos a probar a asignar el antiguo index para que se mantenga aunque se modifique la cardinalidad