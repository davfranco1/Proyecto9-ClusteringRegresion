# Proyecto 9: Clustering y Modelos de Regresión

![imagen](images/header.png)


## Planteamiento del problema 📦📊

- Este proyecto forma parte de un máster de formación en Data Science e Inteligencia Artificial.

- Asumiremos el rol de **científico de datos en una empresa de comercio global**. La compañía busca comprender mejor su base de clientes, productos y operaciones para tomar decisiones informadas que maximicen el beneficio y optimicen sus procesos. 

- Trabajaremos con un conjunto de datos del comercio global que incluye información sobre ventas, envíos, costos y beneficios a nivel de cliente y producto. Nuestra tarea será segmentar los datos mediante **clustering** y luego diseñar modelos de **regresión** específicos para cada segmento, lo que permitirá obtener insights personalizados sobre los factores que influyen en el éxito de la compañía.


## Objetivos del Proyecto ✈️

La empresa tiene las siguientes preguntas clave:

1. **¿Cómo podemos agrupar a los clientes o productos de manera significativa?**

   - Por ejemplo, identificar clientes según su comportamiento de compra o productos según su rentabilidad.

2. **¿Qué factores son más relevantes para predecir el beneficio o las ventas dentro de cada grupo?**

   - Esto ayudará a diseñar estrategias específicas de marketing, optimizar precios o ajustar políticas de descuento.

3. **¿Cómo podemos utilizar estos *insights* para tomar decisiones estratégicas?**

   - Por ejemplo, enfocarse en los segmentos más rentables o intervenir en los menos rentables.

Para contestar estas preguntas, el objetivo en este proyecto es realizar: 

1. **Clustering**: Realizar un análisis de segmentación para agrupar clientes o productos según características clave, las cuales deberás elegir personalmente además de justificar el porque de su elección.

2. **Regresión por Segmentos**: Diseñar modelos de predicción para cada segmento, explicando las relaciones entre variables, intentando predecir el total de ventas en cada uno de los segmentos. 


## Estructura del repositorio

El proyecto está construido de la siguiente manera:

- **datos/**: Carpeta que contiene archivos `.csv`, `.json` o `.pkl` generados durante la captura y tratamiento de los datos.

- **flask/**: Carpeta que contiene un archivo `.py` para la ejecución de la API de *Flask*. Dentro también un Jupyter Notebook para pruebas de la API.

- **images/**: Carpeta que contiene archivos de imagen generados durante la ejecución del código o de fuentes externas.

- **notebooks/**: Carpeta que contiene los archivos `.ipynb` utilizados en el preprocesamiento y modelado de los datos. Dentro, dos carpetas, una para los modelos de clustering y otra para los de regresión. Dentro de "regresión", están numerados para su ejecución secuencial, y contenidos dentro de X carpetas, una para cada modelo, conteniendo cada una de ellas:
  - `1_EDA`
  - `2_Encoding`
  - `3_Outliers`
  - `4_Estandarización`
  - `5_Modelos`

- **src/**: Carpeta que contiene los archivos `.py`, con las funciones y variables utilizadas en los distintos notebooks.

- **streamlit/**: Carpeta que contiene un archivo `.py` para la ejecución de la app *streamlit*.

- **transformers/**: Carpeta que archivos `.pkl` con los objetos de encoding, scaling y el modelo, usados para la transformación de nuevos datos.

- `.gitignore`: Archivo que contiene los archivos y extensiones que no se subirán a nuestro repositorio, como los archivos .env, que contienen contraseñas.


## Lenguaje, librerías y temporalidad
- El proyecto fué elaborado con Python 3.9 y múltiples librerías de soporte:

| **Categoría**                             | **Enlace**                                                                                 |
|-------------------------------------------|-------------------------------------------------------------------------------------------|
| *Librerías para el tratamiento de datos*  | [Pandas](https://pandas.pydata.org/docs/)                                                 |
|                                           | [Numpy](https://numpy.org/doc/)                                                           |
|                                           | [pickle](https://docs.python.org/3/library/pickle.html)                                                           |
|                                           | [json](https://www.w3schools.com/python/python_json.asp)                                                           |
| *Librerías para gestión de APIs*         | [Requests](https://pypi.org/project/requests/)                                            |
| *Librerías para gestión de tiempos*       | [Time](https://docs.python.org/3/library/time.html)                                       |
|                                           | [tqdm](https://numpy.org/doc/)                                                            |
| *Librerías para gráficas*                 | [Plotly](https://plotly.com/python/)                                                      |
|                                           | [Seaborn](https://seaborn.pydata.org)                                                     |
|                                           | [Matplotlib](https://matplotlib.org/stable/index.html)                                    |
|                                           | [shap](https://shap.readthedocs.io/en/latest/)                                            |
| *Librería para controlar parámetros del sistema* | [Sys](https://docs.python.org/3/library/sys.html)                                        |
| *Librería para controlar ficheros*        | [os](https://docs.python.org/3/library/os.html)                                           |
| *Librería para generar aplicaciones basadas en Python* | [streamlit](https://docs.streamlit.io)                                                  |
| *Librería para generar APIs basadas en Python* | [flask](https://flask.palletsprojects.com/en/stable/)                                    |
| *Librería para creación de modelos de Machine Learning* | [scikitlearn](https://scikit-learn.org/stable/)                                         |
| *Librería para la gestión del desbalanceo* | [imblearn](https://imbalanced-learn.org/stable/)                                          |
| *Librería para creación de iteradores (utilizada para combinaciones)* | [itertools](https://docs.python.org/3/library/itertools.html)                           |
| *Librería para la gestión de avisos*      | [warnings](https://docs.python.org/3/library/warnings.html)                               |

- Este proyecto es funcional a fecha 8 de diciembre de 2024.


## Instalación

1. Clona el repositorio
   ```sh
   git clone https://github.com/davfranco1/Proyecto9-ClusteringRegresion.git
   ```

2. Instala las librerías que aparecen en el apartado anterior. Utiliza en tu notebook de Jupyter:
   ```sh
   pip install nombre_librería
   ```

3. Cambia la URL del repositorio remoto para evitar cambios al original.
   ```sh
   git remote set-url origin usuario_github/nombre_repositorio
   git remote -v # Confirma los cambios
   ```

4. Ejecuta el código en los notebooks, modificándolo si es necesario.


## Resultados y conclusiones

- Disponible para descargar [aquí](Resumen.pdf) un PDF resumen con la presentación de los datos, los resultados y conclusiones.


## Próximos pasos

- Crear una plataforma de consulta usando streamlit, que incluya un mapa de calor con los resultados de beneficios por región.
- Dedicar más tiempo a mejorar los resultados de los modelos de regresión.
- Crear un modelo de clustering que no incluya a la variable "Profit".


## Autor

David Franco - [LinkedIn](https://linkedin.com/in/franco-david)

Enlace del proyecto: [https://github.com/davfranco1/Proyecto9-ClusteringRegresion](https://github.com/davfranco1/Proyecto9-ClusteringRegresion)
