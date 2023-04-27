# Procesamiento_DataSets
Repositorio para scripts, tests y procesamiento de los datasets de becas obtenidos del proyecto
Scrap_becas. Etapa 2 de TG.


# Como ejecutar la version estable del motor de recomendación

Usando la consola ejecute el siguiente comando

``` shell
py w2v.py
```

Esto generará algunas recomendaciones usando las 3 métricas de similitud seleccionadas y el modelo
w2v para generar vectores de los requisitos.


# Dev Info

- Python version = 3.10.7
- para generar motor con DT nuevo modelo desde 0 = cleanDT -> extract_keywords -> w2v set RECAL_MODE=True