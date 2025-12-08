# Ajustes de la re-entrega

Tuvimos problemas para importar el archivo "re-alc.py", con lo cual cambiamos el nombre a "re_alc.py". Al resto también le cambiamos "-" por "_" por consistencia.

A continuación, detallamos los cambios realizados en la re-entrega del TP sobre las correcciones realizadas.

Agregamos docstrings en todas las funciones del TP.

## Ejercicio 2
Para este ejercicio notamos que faltaba agregar el parámetro inferior=False en la llamada a la función res_tri_mat para uno de los casos. Por otro lado, agregamos el caso c) para que sea resuelto utilizando el caso a).
Esto lo podemos hacer ya que para la matriz cuadrada de rango completo, la pseudoinversa es lo mismo que la inversa.

## Ejercicio 3
Se corrigió el nombre de pinSVD a pinvSVD.
Ya no asumimos que las matrices S, V y U vienen reducidas, y nos aseguramos de reducirlas en caso de que no lo estén.

## Ejercicio 4
Al igual que en el ejercicio 2, nos faltaba agregar el parámetro inferior=False en la llamada a la función res_tri_mat. Esto nos terminaba arruinando todo el método ya que R es triangular superior.
Ya no asumimos que las matrices Q y R vienen reducidas, y nos aseguramos de reducirlas en caso de que no lo estén.
Notamos que estabamos calculando mal QR utilizando HH, ya que en el proceso creabamos un vector e_1 (canónico), pero no estabamos pidiendo que tenga la misma norma que el vector al cual se va a reflejar.
Aparte de eso, tampoco conservabamos el signo que traía el vector. Ambas cosas están arregladas.

## Notebook
Se corrigió la nomenclatura utilizada de hit-rate por la de accuracy.
Por otro lado, ahora pudimos lograr correr los tres métodos utilizando todo el dataset.

## Aclaraciones extra
Reemplazamos varias funciones básicas como productoMatricial, calcularAx, etc, por sus respectivas versiones de numpy. No utilizamos nada que no se nos halla permitido usar.
Implementamos la función diagRH en su versión iterativa, ya que al correr svd_reducida con el dataset completo estabamos teniendo errores de memoria debido a la gran cantidad de llamadas recursivas.
Como el método svd_reducida tardaba mucho, decidimos optimizarlo y elegir de manera inteligente si trabajar con AAt o AtA dependiendo del caso.
