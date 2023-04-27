# Limitantes 

Como estamos frente un sistema de recomendación basado en contenido, es complicado hallar una metrica para la valoración del modelo de recomendacion implementado. Lo anterior debido a que no se cuenta con un aspecto como lo podria ser el rate de un usuario para cada convocatoria y a partir de ahi hacer calculos que es como la mayoria de las metricas de sistemas de recomendacion basado en usuarios funciona.

Entonces debemos hacer pruebas de conciencia o con usuarios reales que brinden su opinion sobre el funcionamiento del sistema de recomendacion y a partir de ahi obtener una metrica adaptada de las tipicas que se toman en otros modelos como por ejemplo Precision.

# Como evaluar la efectividad del modelo de recomendación 

Usando la metrica de precisión@k la cual el número de items que el usuario cree que son relevantes dividido por el numero de items recomendados.

Para lo anterior vamos a tomar un fracción del dataset por ejemplo el 20 por ciento de los datos y se le pedira a un participante que ejecute el modelo de recomendación usando un seleccion aleatoria de la fraccion del dataset y evalue la relevancia del top 5 de recomendaciones del sistema.

Para determinar la relevancia de las recomendaciones el usuario tendra a su disposicion la lectura de los requisitos y calificara si la recomendacion es relevante o no.

La prueba se realizara con personas que sean del programa de ingenieria de sistemas y se tendra una población reducida de participantes para cumplir con las restricciones temporales del proyecto ya que una prueba masiva tomaria demasiado tiempo.

# Formula de p@k

*p@k* = (# elementos relevantes recomendados) / (# recomendaciones generadas)


*NOTA*: PARA RECALL NO TENEMOS FORMA DE DETERMINAR QUE ELEMENTOS SON RELEVANTES PORQUE NO HAY UN CAMPO DE RATING QUE NOS DIGA CUANDO UN ELEMENTO ES RELEVANTE Y ASI PODER HALLARLO EN LAS RECOMENDACIONES.

