# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

# Otras utilidades
# -----------------------------------------------------------------------
import math
from tqdm import tqdm

# Para las visualizaciones
# -----------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocesado y modelado
# -----------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler

# Sacar número de clusters y métricas
# -----------------------------------------------------------------------
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

# Modelos de clustering
# -----------------------------------------------------------------------
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering

# Para visualizar los dendrogramas
# -----------------------------------------------------------------------
import scipy.cluster.hierarchy as sch

# Para la codificación de las variables numéricas
# -----------------------------------------------------------------------
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder # para poder aplicar los métodos de OneHot, Ordinal,  Label y Target Encoder 
from category_encoders import TargetEncoder, CatBoostEncoder

# Para la gestión de outliers
# -----------------------------------------------------------------------
from itertools import product, combinations
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


class Exploracion:
    """
    Clase para realizar la exploración y visualización de datos en un DataFrame.

    Atributos:
    dataframe : pd.DataFrame
        El conjunto de datos a ser explorado y visualizado.
    """

    def __init__(self, dataframe):
        """
        Inicializa la clase Exploracion con un DataFrame.

        Params:
            - dataframe : pd.DataFrame. El DataFrame que contiene los datos a ser explorados.
        """
        self.dataframe = dataframe
    
    def explorar_datos(self):
        """
        Realiza un análisis exploratorio de un DataFrame.

        Params:
            - Ninguno.

        Returns:
            - None.
        """
        print("5 registros aleatorios:")
        display(self.dataframe.sample(5))
        print("\n")

        print("Información general del DataFrame:")
        print(self.dataframe.info())
        print("\n")

        print("Duplicados en el DataFrame:")
        print(self.dataframe.duplicated().sum())
        print("\n")

        print("Estadísticas descriptivas de las columnas numéricas:")
        display(self.dataframe.describe().T)
        print("\n")

        print("Estadísticas descriptivas de las columnas categóricas:")
        categorical_columns = self.dataframe.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            display(self.dataframe[categorical_columns].describe().T)
        else:
            print("No hay columnas categóricas en el DataFrame.")
        print("\n")
        
        print("Número de valores nulos por columna:")
        print(self.dataframe.isnull().sum())
        print("\n")
        
        if len(categorical_columns) > 0:
            print("Distribución de valores categóricos:")
            for col in categorical_columns:
                print(f"\nColumna: {col}")
                print(self.dataframe[col].value_counts())
        
        print("Matriz de correlación entre variables numéricas:")
        display(self.dataframe.corr(numeric_only=True))
        print("\n")

    def visualizar_numericas(self):
        """
        Genera histogramas, boxplots y gráficos de dispersión para las variables numéricas del DataFrame.

        Params:
            - Ninguno.

        Returns:
            - None.
        """
        columns = self.dataframe.select_dtypes(include=np.number).columns

        # Histogramas
        fig, axes = plt.subplots(nrows=math.ceil(len(columns)/2), ncols=2, figsize=(21, 13))
        axes = axes.flat
        plt.suptitle("Distribución de las variables numéricas", fontsize=24)
        for indice, columna in enumerate(columns):
            sns.histplot(x=columna, data=self.dataframe, ax=axes[indice], kde=True, color="#F2C349")

        if len(columns) % 2 != 0:
            fig.delaxes(axes[-1])

        plt.tight_layout()

        # Boxplots
        fig, axes = plt.subplots(nrows=math.ceil(len(columns)/2), ncols=2, figsize=(19, 11))
        axes = axes.flat
        plt.suptitle("Boxplots de las variables numéricas", fontsize=24)
        for indice, columna in enumerate(columns):
            sns.boxplot(x=columna, data=self.dataframe, ax=axes[indice], color="#F2C349", flierprops={'markersize': 4, 'markerfacecolor': 'cyan'})
        if len(columns) % 2 != 0:
            fig.delaxes(axes[-1])
        plt.tight_layout()
    
    def visualizar_categoricas(self):
        """
        Genera gráficos de barras (count plots) para las variables categóricas del DataFrame.

        Params:
            - Ninguno.

        Returns:
            - None.
        """
        categorical_columns = self.dataframe.select_dtypes(include=['object', 'category']).columns

        if len(categorical_columns) > 0:
            try:
                _, axes = plt.subplots(nrows=len(categorical_columns), ncols=1, figsize=(15, 5 * len(categorical_columns)))
                axes = axes.flat
                plt.suptitle("Distribución de las variables categóricas", fontsize=24)
                for indice, columna in enumerate(categorical_columns):
                    sns.countplot(data=self.dataframe, x=columna, ax=axes[indice])
                    axes[indice].set_title(f'Distribución de {columna}', fontsize=20)
                    axes[indice].set_xlabel(columna, fontsize=16)
                    axes[indice].set_ylabel('Conteo', fontsize=16)
                plt.tight_layout()
            except: 
                sns.countplot(data=self.dataframe, x=categorical_columns[0])
                plt.title(f'Distribución de {categorical_columns[0]}', fontsize=20)
                plt.xlabel(categorical_columns[0], fontsize=16)
                plt.ylabel('Conteo', fontsize=16)
        else:
            print("No hay columnas categóricas en el DataFrame.")

    def visualizar_categoricas_numericas(self):
        """
        Genera gráficos de dispersión para las variables numéricas vs todas las variables categóricas.

        Params:
            - Ninguno.

        Returns:
            - None.
        """
        categorical_columns = self.dataframe.select_dtypes(include=['object', 'category']).columns
        numerical_columns = self.dataframe.select_dtypes(include=np.number).columns
        if len(categorical_columns) > 0:
            for num_col in numerical_columns:
                try:
                    _, axes = plt.subplots(nrows=len(categorical_columns), ncols=1, figsize=(10, 5 * len(categorical_columns)))
                    axes = axes.flat
                    plt.suptitle(f'Dispersión {num_col} vs variables categóricas', fontsize=24)
                    for indice, cat_col in enumerate(categorical_columns):
                        sns.scatterplot(x=num_col, y=self.dataframe.index, hue=cat_col, data=self.dataframe, ax=axes[indice])
                        axes[indice].set_xlabel(num_col, fontsize=16)
                        axes[indice].set_ylabel('Índice', fontsize=16)
                        axes[indice].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
                    plt.tight_layout()
                except: 
                    sns.scatterplot(x=num_col, y=self.dataframe.index, hue=categorical_columns[0], data=self.dataframe)
                    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=10)
                    plt.xlabel(num_col, fontsize=16)
                    plt.ylabel('Índice', fontsize=16)
        else:
            print("No hay columnas categóricas en el DataFrame.")

    def correlacion(self, metodo="pearson", tamanio=(14, 8)):
        """
        Genera un heatmap de la matriz de correlación de las variables numéricas del DataFrame.

        Params:
            - metodo : str, optional, default: "pearson". Método para calcular la correlación.
            - tamanio : tuple of int, optional, default: (14, 8). Tamaño de la figura del heatmap.

        Returns:
            - None.
        """
        plt.figure(figsize=tamanio)
        mask = np.triu(np.ones_like(self.dataframe.corr(numeric_only=True), dtype=np.bool_))
        sns.heatmap(self.dataframe.corr(numeric_only=True, method=metodo), annot=True, cmap='viridis', vmax=1, vmin=-1, mask=mask)
        plt.title("Correlación de las variables numéricas", fontsize=24)


class Preprocesado:
    """
    Clase para realizar preprocesamiento de datos en un DataFrame.

    Atributos:
        - dataframe : pd.DataFrame. El conjunto de datos a ser preprocesado.
    """
    
    def __init__(self, dataframe):
        """
        Inicializa la clase Preprocesado con un DataFrame.

        Params:
            - dataframe : pd.DataFrame. El DataFrame que contiene los datos a ser preprocesados.
        """
        self.dataframe = dataframe

    def estandarizar(self):
        """
        Estandariza las columnas numéricas del DataFrame.

        Este método ajusta y transforma las columnas numéricas del DataFrame utilizando `StandardScaler` para que
        tengan media 0 y desviación estándar 1.

        Returns:
            - pd.DataFrame. El DataFrame con las columnas numéricas estandarizadas.
        """
        # Sacamos el nombre de las columnas numéricas
        col_numericas = self.dataframe.select_dtypes(include=np.number).columns

        # Inicializamos el escalador para estandarizar los datos
        scaler = StandardScaler()

        # Ajustamos los datos y los transformamos
        X_scaled = scaler.fit_transform(self.dataframe[col_numericas])

        # Sobreescribimos los valores de las columnas en el DataFrame
        self.dataframe[col_numericas] = X_scaled

        return self.dataframe
    
    def codificar(self):
        """
        Codifica las columnas categóricas del DataFrame.

        Este método reemplaza los valores de las columnas categóricas por sus frecuencias relativas dentro de cada
        columna.

        Returns:
            - pd.DataFrame. El DataFrame con las columnas categóricas codificadas.
        """
        # Sacamos el nombre de las columnas categóricas
        col_categoricas = self.dataframe.select_dtypes(include=["category", "object"]).columns

        # Iteramos por cada una de las columnas categóricas para aplicar el encoding
        for categoria in col_categoricas:
            # Calculamos las frecuencias de cada una de las categorías
            frecuencia = self.dataframe[categoria].value_counts(normalize=True)

            # Mapeamos los valores obtenidos en el paso anterior, sobreescribiendo la columna original
            self.dataframe[categoria] = self.dataframe[categoria].map(frecuencia)

        return self.dataframe


class Clustering:
    """
    Clase para realizar varios métodos de clustering en un DataFrame.

    Atributos:
        - dataframe : pd.DataFrame. El conjunto de datos sobre el cual se aplicarán los métodos de clustering.
    """
    
    def __init__(self, dataframe):
        """
        Inicializa la clase Clustering con un DataFrame.

        Params:
            - dataframe : pd.DataFrame. El DataFrame que contiene los datos a los que se les aplicarán los métodos de clustering.
        """
        self.dataframe = dataframe
    
    def sacar_clusters_kmeans(self, n_clusters=(2, 15)):
        """
        Utiliza KMeans y KElbowVisualizer para determinar el número óptimo de clusters basado en la métrica de silhouette.

        Params:
            - n_clusters : tuple of int, optional, default: (2, 15). Rango de número de clusters a probar.
        
        Returns:
            None
        """
        model = KMeans(random_state=42)
        visualizer = KElbowVisualizer(model, k=n_clusters, metric='silhouette')
        visualizer.fit(self.dataframe)
        visualizer.show()
    
    def modelo_kmeans(self, dataframe_original, num_clusters):
        """
        Aplica KMeans al DataFrame y añade las etiquetas de clusters al DataFrame original.

        Params:
            - dataframe_original : pd.DataFrame. El DataFrame original al que se le añadirán las etiquetas de clusters.
            - num_clusters : int. Número de clusters a formar.

        Returns:
            - pd.DataFrame. El DataFrame original con una nueva columna para las etiquetas de clusters.
        """
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        km_fit = kmeans.fit(self.dataframe)
        labels = km_fit.labels_
        dataframe_original["clusters_kmeans"] = labels.astype(str)
        return dataframe_original, labels, kmeans
    
    def visualizar_dendrogramas(self, lista_metodos=["average", "complete", "ward", "single"]):
        """
        Genera y visualiza dendrogramas para el conjunto de datos utilizando diferentes métodos de distancias.

        Params:
            - lista_metodos : list of str, optional, default: ["average", "complete", "ward", "single"]. Lista de métodos para calcular las distancias entre los clusters. Cada método generará un dendrograma
                en un subplot diferente.

        Returns:
            None
        """
        _, axes = plt.subplots(nrows=1, ncols=len(lista_metodos), figsize=(20, 8))
        axes = axes.flat

        for indice, metodo in enumerate(lista_metodos):
            sch.dendrogram(sch.linkage(self.dataframe, method=metodo),
                           labels=self.dataframe.index, 
                           leaf_rotation=90, leaf_font_size=4,
                           ax=axes[indice])
            axes[indice].set_title(f'Dendrograma usando {metodo}')
            axes[indice].set_xlabel('Muestras')
            axes[indice].set_ylabel('Distancias')
    
    def modelo_aglomerativo(self, num_clusters, metodo_distancias, dataframe_original):
        """
        Aplica clustering aglomerativo al DataFrame y añade las etiquetas de clusters al DataFrame original.

        Params:
            - num_clusters : int. Número de clusters a formar.
            - metodo_distancias : str. Método para calcular las distancias entre los clusters.
            - dataframe_original : pd.DataFrame. El DataFrame original al que se le añadirán las etiquetas de clusters.

        Returns:
            - pd.DataFrame. El DataFrame original con una nueva columna para las etiquetas de clusters.
        """
        modelo = AgglomerativeClustering(
            linkage=metodo_distancias,
            distance_threshold=None,
            n_clusters=num_clusters
        )
        aglo_fit = modelo.fit(self.dataframe)
        labels = aglo_fit.labels_
        dataframe_original["clusters_agglomerative"] = labels.astype(str)
        return dataframe_original
    
    def modelo_divisivo(self, dataframe_original, threshold=0.5, max_clusters=5):
        """
        Implementa el clustering jerárquico divisivo.

        Params:
            - dataframe_original : pd.DataFrame. El DataFrame original al que se le añadirán las etiquetas de clusters.
            - threshold : float, optional, default: 0.5. Umbral para decidir cuándo dividir un cluster.
            - max_clusters : int, optional, default: 5. Número máximo de clusters deseados.

        Returns:
            - pd.DataFrame. El DataFrame original con una nueva columna para las etiquetas de los clusters.
        """
        def divisive_clustering(data, current_cluster, cluster_labels):
            # Si el número de clusters actuales es mayor o igual al máximo permitido, detener la división
            if len(set(current_cluster)) >= max_clusters:
                return current_cluster

            # Aplicar KMeans con 2 clusters
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(data)
            labels = kmeans.labels_

            # Calcular la métrica de silueta para evaluar la calidad del clustering
            silhouette_avg = silhouette_score(data, labels)

            # Si la calidad del clustering es menor que el umbral o si el número de clusters excede el máximo, detener la división
            if silhouette_avg < threshold or len(set(current_cluster)) + 1 > max_clusters:
                return current_cluster

            # Crear nuevas etiquetas de clusters
            new_cluster_labels = current_cluster.copy()
            max_label = max(current_cluster)

            # Asignar nuevas etiquetas incrementadas para cada subcluster
            for label in set(labels):
                cluster_indices = np.where(labels == label)[0]
                new_label = max_label + 1 + label
                new_cluster_labels[cluster_indices] = new_label

            # Aplicar recursión para seguir dividiendo los subclusters
            for new_label in set(new_cluster_labels):
                cluster_indices = np.where(new_cluster_labels == new_label)[0]
                new_cluster_labels = divisive_clustering(data[cluster_indices], new_cluster_labels, new_cluster_labels)

            return new_cluster_labels

        # Inicializar las etiquetas de clusters con ceros
        initial_labels = np.zeros(len(self.dataframe))

        # Llamar a la función recursiva para iniciar el clustering divisivo
        final_labels = divisive_clustering(self.dataframe.values, initial_labels, initial_labels)

        # Añadir las etiquetas de clusters al DataFrame original
        dataframe_original["clusters_divisive"] = final_labels.astype(int).astype(str)

        return dataframe_original

    def modelo_espectral(self, dataframe_original, n_clusters=3, assign_labels='kmeans'):
        """
        Aplica clustering espectral al DataFrame y añade las etiquetas de clusters al DataFrame original.

        Params:
            - dataframe_original : pd.DataFrame. El DataFrame original al que se le añadirán las etiquetas de clusters.
            - n_clusters : int, optional, default: 3. Número de clusters a formar.
            - assign_labels : str, optional, default: 'kmeans'. Método para asignar etiquetas a los puntos. Puede ser 'kmeans' o 'discretize'.

        Returns:
            - pd.DataFrame. El DataFrame original con una nueva columna para las etiquetas de clusters.
        """
        spectral = SpectralClustering(n_clusters=n_clusters, assign_labels=assign_labels, random_state=0)
        labels = spectral.fit_predict(self.dataframe)
        dataframe_original["clusters_spectral"] = labels.astype(str)
        return dataframe_original
    
    def modelo_dbscan(self, dataframe_original, eps_values=[0.5, 1.0, 1.5], min_samples_values=[3, 2, 1]):
        """
        Aplica DBSCAN al DataFrame y genera clusters con diferentes combinaciones de parámetros. 
        Evalúa las métricas de calidad y retorna el DataFrame original con etiquetas de clusters.

        Parámetros:
        -----------
        dataframe_original : pd.DataFrame
            DataFrame original al que se le añadirán las etiquetas de clusters.
        eps_values : list of float, optional, default: [0.5, 1.0, 1.5]
            Lista de valores para el parámetro eps de DBSCAN.
        min_samples_values : list of int, optional, default: [3, 2, 1]
            Lista de valores para el parámetro min_samples de DBSCAN.

        Retorna:
        --------
        dataframe_original : pd.DataFrame
            DataFrame con una nueva columna `clusters_dbscan` que contiene las etiquetas de los clusters.
        metrics_df_dbscan : pd.DataFrame
            DataFrame con las métricas de evaluación para cada combinación de parámetros.
        """
        # Validación de entradas
        if not eps_values or not min_samples_values:
            raise ValueError("Las listas eps_values y min_samples_values no pueden estar vacías.")

        # Inicializar variables para almacenar el mejor modelo
        best_eps = None
        best_min_samples = None
        best_silhouette = float("-inf")
        metrics_results_dbscan = []

        # Iterar sobre combinaciones de parámetros
        for eps in tqdm(eps_values, desc=f"Iterando sobre eps y min_samples"):
            for min_samples in min_samples_values:
                try:
                    # Aplicar DBSCAN
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(dataframe_original)

                    # Evaluar solo si hay más de un cluster
                    if len(set(labels)) > 1 and len(set(labels)) < len(labels):
                        silhouette = silhouette_score(dataframe_original, labels)
                        davies_bouldin = davies_bouldin_score(dataframe_original, labels)

                        # Cardinalidad de los clusters
                        unique, counts = np.unique(labels, return_counts=True)
                        cardinalidad = dict(zip(unique, counts))

                        # Guardar resultados
                        metrics_results_dbscan.append({
                            "eps": eps,
                            "min_samples": min_samples,
                            "silhouette_score": silhouette,
                            "davies_bouldin_score": davies_bouldin,
                            "cardinality": cardinalidad
                        })

                        # Actualizar el mejor modelo
                        if silhouette > best_silhouette:
                            best_silhouette = silhouette
                            best_eps = eps
                            best_min_samples = min_samples
                except Exception as e:
                    print(f"Error con eps={eps}, min_samples={min_samples}: {e}")

        # Crear un DataFrame con las métricas
        metrics_df_dbscan = pd.DataFrame(metrics_results_dbscan).sort_values(by="silhouette_score", ascending=False)

        # Mostrar resultados
        display(metrics_df_dbscan)

        # Aplicar DBSCAN con los mejores parámetros
        if best_eps is not None and best_min_samples is not None:
            best_dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
            dataframe_original["clusters_dbscan"] = best_dbscan.fit_predict(dataframe_original)
            print(f"Mejor modelo: eps={best_eps}, min_samples={best_min_samples}, silhouette_score={best_silhouette:.4f}")
        else:
            print("No se encontraron clusters válidos con los parámetros proporcionados.")
            dataframe_original["clusters_dbscan"] = -1  # Asignar ruido si no se encontraron clusters

        cantidad_clusters = dataframe_original["clusters_dbscan"].nunique()
        print(f"Se han generado {cantidad_clusters} clusters (incluyendo outliers, si los hay).")

        return dataframe_original, metrics_df_dbscan


    def plot_epsilon(self, dataframe, k=10, figsize=(10, 6), ylim=(0, 0.2)):
        """
        Genera un gráfico de k-distancias para determinar el valor óptimo de epsilon (eps)
        al usar el algoritmo DBSCAN, y calcula numéricamente el valor de epsilon basado 
        en el máximo gradiente (codo).

        Parámetros:
        -----------
        dataframe : pd.DataFrame o np.ndarray
            Conjunto de datos a analizar. Cada fila es un punto, y las columnas son las características.
        k : int, opcional, default=10
            Número de vecinos a considerar. Generalmente se recomienda usar un valor igual al
            número de variables por 2, o ligeramente mayor.
        figsize : tuple, opcional, default=(10, 6)
            Tamaño de la figura del gráfico (ancho, alto).
        ylim : tuple, opcional, default=(0, 0.2)
            Límite en el eje y para enfocar la visualización de las distancias.

        Salida:
        -------
        Gráfico de k-distancias:
            - El eje x representa los puntos ordenados por distancia al k-ésimo vecino más cercano.
            - El eje y muestra la distancia al k-ésimo vecino más cercano.
            - La línea vertical roja indica el epsilon recomendado basado en el codo.

        Retorno:
        --------
        eps_value : float
            Valor recomendado para epsilon (eps).

        Notas:
        ------
        - Es importante escalar los datos antes de usar esta función, especialmente si las características 
        tienen diferentes rangos.

        Ejemplo:
        --------
        # Calcular y graficar distancias para encontrar eps
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(dataframe)
        self.plot_epsilon(dataframe=scaled_data, k=10, figsize=(12, 8), ylim=(0, 0.5))
        """
        # Paso 1: Encontrar a los vecinos más cercanos
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(dataframe)
        
        # Paso 2: calcular las distancias entre vecinos
        distances, indices = neighbors_fit.kneighbors(dataframe)
        k_distances = distances[:, k - 1]  # k-th nearest neighbor distances

        # Paso 3: Ordenar distancias
        k_distances = np.sort(k_distances)
        
        # Paso 4: Calcular el máximo gradiente (codo)
        gradients = np.diff(k_distances)  # Derivada numérica
        max_gradient_index = np.argmax(gradients)  # Índice con el mayor gradiente
        eps_value = k_distances[max_gradient_index]  # Valor de epsilon recomendado

        # Paso 5: Graficar el k-distance plot
        plt.figure(figsize=figsize)
        plt.plot(k_distances, label="distancias 'k'")
        plt.axvline(x=max_gradient_index, color='red', linestyle='--', label=f"Epsilon = {eps_value:.4f}")
        plt.ylim(ylim)
        plt.title(f"Plot K-Distance con k={k}")
        plt.xlabel("Puntos ordenados por distancia")
        plt.ylabel(f"Distancia al {k}-ésimo vecino")
        plt.legend()
        plt.grid()
        plt.show()

        # Retornar el valor de epsilon
        return eps_value

    def calcular_metricas(self, labels: np.ndarray):
        """
        Calcula métricas de evaluación del clustering.
        """
        if len(set(labels)) <= 1:
            raise ValueError("El clustering debe tener al menos 2 clusters para calcular las métricas.")

        silhouette = silhouette_score(self.dataframe, labels)
        davies_bouldin = davies_bouldin_score(self.dataframe, labels)

        unique, counts = np.unique(labels, return_counts=True)
        cardinalidad = dict(zip(unique, counts))

        return pd.DataFrame({
            "silhouette_score": silhouette,
            "davies_bouldin_index": davies_bouldin,
            "cardinalidad": [cardinalidad]
        }, index = [0])
    
    def plot_clusters(self, df_cluster, columna_cluster):
        columnas_plot = df_cluster.columns.drop(columna_cluster)

        ncols = math.ceil(len(columnas_plot) / 2)
        nrows = 2

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 8))
        axes = axes.flat

        for indice, columna in enumerate(columnas_plot):
            df_group = df_cluster.groupby(columna_cluster)[columna].mean().reset_index()
            sns.barplot(x=columna_cluster, y=columna, data=df_group, ax=axes[indice], palette="coolwarm")
            axes[indice].set_title(columna)  

        if len(columnas_plot) % 2 == 1: 
            fig.delaxes(axes[-1]) 

        plt.tight_layout()
        plt.show() 


    def radar_plot(self, dataframe, variables, columna_cluster):
        """
        Genera un gráfico radar (radar plot) para visualizar las características promedio 
        de cada cluster.

        Parámetros:
        -----------
        dataframe : pd.DataFrame
            DataFrame que contiene las variables a analizar y la columna con la etiqueta del cluster.
        variables : list
            Lista de nombres de las columnas que se incluirán en el radar plot. Estas deben ser 
            variables numéricas.
        columna_cluster : str
            Nombre de la columna que contiene las etiquetas de los clusters.

        Salida:
        -------
        Gráfico radar:
            - Cada eje representa una de las variables seleccionadas.
            - Las líneas y áreas sombreadas corresponden a las medias de cada cluster en las variables.

        Notas:
        ------
        - Las variables seleccionadas deben ser numéricas, ya que se calcula la media de cada 
        cluster para graficarlas.
        - El gráfico cierra el radar repitiendo la primera variable al final para que las líneas 
        formen un ciclo completo.

        Ejemplo:
        --------
        # Generar radar plot para clusters en el DataFrame
        variables = ['edad', 'ingresos', 'gasto']
        self.radar_plot(dataframe=df, variables=variables, columna_cluster='clusters_dbscan')
        """
        # Agrupar por cluster y calcular la media
        cluster_means = dataframe.groupby(columna_cluster)[variables].mean()

        # Repetir la primera columna al final para cerrar el radar
        cluster_means = pd.concat([cluster_means, cluster_means.iloc[:, 0:1]], axis=1)

        # Crear los ángulos para el radar plot
        num_vars = len(variables)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Cerrar el gráfico

        # Crear el radar plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # Dibujar un gráfico para cada cluster
        for i, row in cluster_means.iterrows():
            ax.plot(angles, row, label=f'Cluster {i}')
            ax.fill(angles, row, alpha=0.25)

        # Configurar etiquetas de los ejes
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(variables)

        # Añadir leyenda y título
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title('Radar Plot de los Clusters', size=16)
        plt.show()


    def plot_completo_clusters(self, data, x, y=None, dateplot="month", hue=None):
        """
        Genera múltiples gráficos de las columnas seleccionadas de un DataFrame, con la opción 
        de usar una variable como "hue" para diferenciar grupos o categorías.

        Créditos:
        -----------
        Como base, he tomado una función del chino murciano más majo del mundo. Gracias <3 @https://github.com/yanruwu

        Parámetros:
        -----------
        data : pd.DataFrame
            El DataFrame que contiene los datos para graficar.
        x : str o list
            Columnas del DataFrame a graficar. Puede ser un string (columna única) o una lista 
            de nombres de columnas.
        y : str, opcional
            Columna del DataFrame para usar como variable dependiente en los gráficos (para gráficos
            como scatterplot, boxplot, etc.). Si no se especifica, se grafican las columnas de `x` 
            individualmente.
        dateplot : {"month", "year"}, opcional, default="month"
            Especifica cómo manejar las columnas de tipo fecha. Si es "month", las fechas se 
            agruparán por mes. Si es "year", se agruparán por año.
        hue : str, opcional
            Columna del DataFrame que se usará como "hue" para diferenciar colores en los gráficos. 
            Esto es útil para comparar grupos o clusters.

        Salida:
        -------
        Gráficos generados:
            - Para columnas numéricas: histogramas con la opción de kde (curva de densidad).
            - Para columnas categóricas: gráficos de conteo (countplot).
            - Para columnas de fecha: gráficos de conteo agrupados por mes o año.
            - Para combinaciones de columnas con `y`:
                * Numérico vs Numérico: scatterplot.
                * Categórico vs Numérico: boxplot.
                * Fecha vs Numérico: lineplot.

        Notas:
        ------
        - Si no se especifica `y`, la función genera gráficos individuales para cada columna en `x`.
        - Si se especifica `hue`, los gráficos mostrarán los datos diferenciados por la categoría 
        especificada en `hue`.

        Ejemplo:
        --------
        # Graficar todas las columnas con los clusters como hue
        plot_completo_clusters(data=df, x=df.columns, hue="clusters_dbscan")

        # Graficar columnas específicas
        plot_completo_clusters(data=df, x=["edad", "ingresos"], y="gasto", hue="segmento")

        """

        if type(x) == str:
            x = [x] #convertir x en lista

        # Separar tipos de datos
        num_cols = data[x].select_dtypes("number")
        cat_cols = data[x].select_dtypes("O", "category")
        date_cols = data[x].select_dtypes("datetime")

        if not y:
            nrows = 2
            ncols = math.ceil(len(x) / 2)
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, 15), dpi=130)
            axes = axes.flat
            for i, col in enumerate(x):
                if col in num_cols:
                    sns.histplot(data=data, x=col, hue=hue, ax=axes[i], bins=20, kde=True)
                    axes[i].set_title(col)
                elif col in cat_cols:
                    sns.countplot(data=data, x=col, hue=hue, ax=axes[i])
                    axes[i].tick_params(axis='x', rotation=90)
                    axes[i].set_title(col)
                elif col in date_cols:
                    sns.countplot(
                        data=data,
                        x=data[col].apply(lambda x: x.year) if dateplot == "year" else data[col].apply(lambda x: x.month),
                        hue=hue,
                        ax=axes[i]
                    )
                    axes[i].tick_params(axis='x', rotation=90)
                    axes[i].set_title(f"{col}_{dateplot}")
                else:
                    print(f"Advertencia: No se pudo graficar '{col}' ya que no es de un tipo soportado o la combinación no es válida.")
                
                axes[i].set_xlabel("")
                
            for j in range(len(x), len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.show()

        else:
            nrows = math.ceil(len(x) / 2)
            ncols = 2
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 25), dpi=130)
            axes = axes.flat

            for i, col in enumerate(x):
                # Caso 1: Numérico vs Numérico -> Scatterplot
                if col in num_cols and y in data.select_dtypes("number"):
                    sns.scatterplot(data=data, x=col, y=y, hue=hue, ax=axes[i])
                    axes[i].set_title(f'{col} vs {y}')
                # Caso 2: Categórico vs Numérico -> Boxplot o Violinplot
                elif col in cat_cols and y in data.select_dtypes("number"):
                    sns.boxplot(data=data, x=col, y=y, hue=hue, ax=axes[i])
                    axes[i].tick_params(axis='x', rotation=90)
                    axes[i].set_title(f'{col} vs {y}')
                # Caso 3: Fecha vs Numérico -> Lineplot
                elif col in date_cols and y in data.select_dtypes("number"):
                    date_data = data[col].dt.year if dateplot == "year" else data[col].dt.month
                    sns.lineplot(x=date_data, y=data[y], hue=hue, ax=axes[i])
                    axes[i].set_title(f'{col}_{dateplot} vs {y}')
                    axes[i].tick_params(axis='x', rotation=90)
                else:
                    print(f"Advertencia: No se pudo graficar '{col}' frente a '{y}' ya que no es de un tipo soportado o la combinación no es válida.")

                axes[i].set_xlabel("")
            
            for j in range(len(x), len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.show()


    def modelo_aglomerativo_tabla(self, dataframe, linkage_methods, distance_metrics):
        """
        Realiza clustering jerárquico (AgglomerativeClustering) utilizando diferentes métodos de enlace 
        y métricas de distancia, y evalúa los resultados con varias métricas de validación.

        Parámetros:
        -----------
        dataframe : pd.DataFrame o np.ndarray
            Conjunto de datos a analizar. Se recomienda que los datos estén preprocesados y normalizados.
        linkage_methods : list
            Lista de métodos de enlace a usar en AgglomerativeClustering. Ejemplos: ["ward", "complete", "average", "single"].
        distance_metrics : list
            Lista de métricas de distancia para calcular la proximidad entre puntos. Ejemplos: ["euclidean", "manhattan", "cosine"].

        Salida:
        -------
        results_df : pd.DataFrame
            DataFrame que contiene los resultados de las configuraciones probadas, incluyendo:
            - linkage: Método de enlace utilizado.
            - metric: Métrica de distancia utilizada.
            - silhouette_score: Promedio del puntaje de silueta para los clusters.
            - davies_bouldin_index: Índice Davies-Bouldin, donde valores más bajos indican mejor separación de clusters.
            - cluster_cardinality: Diccionario con la cardinalidad (tamaño) de cada cluster.
            - n_cluster: Número de clusters usados.

        Notas:
        ------
        - El modelo se ajusta para diferentes combinaciones de métodos de enlace y métricas de distancia.
        - Si ocurre un error (por ejemplo, usar "ward" con una métrica no euclidiana), se captura y muestra.
        - Solo se calculan métricas de validación si hay más de un cluster.
        - Ordena los resultados por `silhouette_score` de mayor a menor.

        Ejemplo:
        --------
        linkage_methods = ["ward", "complete", "average"]
        distance_metrics = ["euclidean", "manhattan"]
        resultados = modelo_aglomerativo_tabla(dataframe=df_normalizado, linkage_methods=linkage_methods, distance_metrics=distance_metrics)

        # Ver los mejores resultados
        print(resultados.head())
        """

        # Crear un DataFrame para almacenar los resultados
        results = []

        # Suponiendo que tienes un DataFrame llamado df_copia
        # Aquí df_copia debería ser tu conjunto de datos
        # Asegúrate de que esté preprocesado adecuadamente (normalizado si es necesario)

        for linkage_method in linkage_methods:
            for metric in distance_metrics:
                for cluster in range(2,6):
                    try:
                        # Configurar el modelo de AgglomerativeClustering
                        modelo = AgglomerativeClustering(
                            linkage=linkage_method,
                            metric=metric,  
                            distance_threshold=None,  # Para buscar n_clusters
                            n_clusters=cluster, # Cambia esto según tu análisis. Si tienen valor, el distance threshold puede ser none, y viceversa.
                        )
                        
                        # Ajustar el modelo
                        labels = modelo.fit_predict(dataframe) #etiquetas del cluster al que pertenece

                        # Calcular métricas si hay más de un cluster
                        if len(np.unique(labels)) > 1:
                            # Silhouette Score
                            silhouette_avg = silhouette_score(dataframe, labels, metric=metric)

                            # Davies-Bouldin Index
                            db_score = davies_bouldin_score(dataframe, labels)

                            
                            # Cardinalidad (tamaño de cada cluster)
                            cluster_cardinality = {cluster: sum(labels == cluster) for cluster in np.unique(labels)}
                        else:
                            cluster_cardinality = {'Cluster único': len(dataframe)}

                        # Almacenar resultados
                        results.append({
                            'linkage': linkage_method,
                            'metric': metric,
                            'silhouette_score': silhouette_avg,
                            'davies_bouldin_index': db_score,
                            'cluster_cardinality': cluster_cardinality,
                            'n_cluster': cluster
                        })

                    except Exception as e:
                        print(f"Error con linkage={linkage_method}, metric={metric}: {e}")
                        # Ward SÓLO acepta medir con distancia euclidiana en scikitlearn.

        # Crear DataFrame de resultados
        results_df = pd.DataFrame(results)

        # Mostrar resultados ordenados por silhouette_score
        results_df = results_df.sort_values(by='silhouette_score', ascending=False)

        # Mostrar el DataFrame
        return results_df


class Encoding:
    """
    Clase para realizar diferentes tipos de codificación en un DataFrame.

    Parámetros:
        - dataframe: DataFrame de pandas, el conjunto de datos a codificar.
        - diccionario_encoding: dict, un diccionario que especifica los tipos de codificación a realizar.
        - variable_respuesta: str, el nombre de la variable objetivo.

    Métodos:
        - one_hot_encoding(): Realiza codificación one-hot en las columnas especificadas en el diccionario de codificación.
        - get_dummies(): Realiza codificación get_dummies en las columnas especificadas en el diccionario de codificación.
        - ordinal_encoding(): Realiza codificación ordinal en las columnas especificadas en el diccionario de codificación.
        - label_encoding(): Realiza codificación label en las columnas especificadas en el diccionario de codificación.
        - target_encoding(): Realiza codificación target en la variable especificada en el diccionario de codificación.
        - frequency_encoding(): Realiza codificación de frecuencia en las columnas especificadas en el diccionario de codificación.
        - catboost_encoding(): Realiza codificación CatBoost en las columnas especificadas en el diccionario de codificación.
    """

    def __init__(self, dataframe, diccionario_encoding, variable_respuesta):
        self.dataframe = dataframe
        self.diccionario_encoding = diccionario_encoding
        self.variable_respuesta = variable_respuesta

    def one_hot_encoding(self):
        col_encode = self.diccionario_encoding.get("onehot", [])
        if col_encode:
            one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            trans_one_hot = one_hot_encoder.fit_transform(self.dataframe[col_encode])
            oh_df = pd.DataFrame(trans_one_hot, columns=one_hot_encoder.get_feature_names_out(col_encode))
            oh_df.index = self.dataframe.index
            self.dataframe = pd.concat([self.dataframe.drop(columns=col_encode), oh_df], axis=1)
        return self.dataframe, one_hot_encoder

    def get_dummies(self, prefix='category', prefix_sep="_"):
        col_encode = self.diccionario_encoding.get("dummies", [])
        if col_encode:
            df_dummies = pd.get_dummies(self.dataframe[col_encode], dtype=int, prefix=prefix, prefix_sep=prefix_sep)
            self.dataframe = pd.concat([self.dataframe.reset_index(drop=True), df_dummies.reset_index(drop=True)], axis=1)
            self.dataframe.drop(col_encode, axis=1, inplace=True)
        return self.dataframe

    def ordinal_encoding(self):
        col_encode = self.diccionario_encoding.get("ordinal", {})
        if col_encode:
            orden_categorias = list(self.diccionario_encoding["ordinal"].values())
            ordinal_encoder = OrdinalEncoder(categories=orden_categorias, dtype=float, handle_unknown="use_encoded_value", unknown_value=np.nan)
            ordinal_encoder_trans = ordinal_encoder.fit_transform(self.dataframe[col_encode.keys()])
            self.dataframe.drop(col_encode, axis=1, inplace=True)
            ordinal_encoder_df = pd.DataFrame(ordinal_encoder_trans, columns=ordinal_encoder.get_feature_names_out())
            self.dataframe = pd.concat([self.dataframe.reset_index(drop=True), ordinal_encoder_df], axis=1)
        return self.dataframe

    def label_encoding(self):
        col_encode = self.diccionario_encoding.get("label", [])
        if col_encode:
            label_encoder = LabelEncoder()
            self.dataframe[col_encode] = self.dataframe[col_encode].apply(lambda col: label_encoder.fit_transform(col))
        return self.dataframe
    
    def target_encoding(self):
        """
        Realiza codificación target en la variable especificada en el diccionario de codificación.

        Returns:
        
        dataframe: DataFrame de pandas, el DataFrame con codificación target aplicada."""

        df_sin_vr = self.dataframe.copy()
        df_sin_vr.drop(columns=[f"{self.variable_respuesta}"], inplace=True)

        #accedemos a la clave de 'target' para poder extraer las columnas a las que que queramos aplicar Target Encoding. En caso de que no exista la clave, esta variable será una lista vacía
        col_encode = self.diccionario_encoding.get("target", [])

        # si hay contenido en la lista 
        if col_encode:

            target_encoder = TargetEncoder(cols=col_encode)
            df_target = target_encoder.fit_transform(df_sin_vr, self.dataframe[self.variable_respuesta])
            self.dataframe = pd.concat([self.dataframe[self.variable_respuesta].reset_index(drop=True), df_target], axis=1)

        return self.dataframe, target_encoder

    def frequency_encoding(self):
        col_encode = self.diccionario_encoding.get("frequency", [])
        if col_encode:
            for categoria in col_encode:
                frecuencia = self.dataframe[categoria].value_counts(normalize=True)
                self.dataframe[categoria] = self.dataframe[categoria].map(frecuencia)
        return self.dataframe

    def catboost_encoding(self):
        col_encode = self.diccionario_encoding.get("catboost", [])

        if col_encode:
            encoder = CatBoostEncoder()
            
            for col in col_encode:
                if col != self.variable_respuesta:
                    self.dataframe[col] = encoder.fit_transform(
                        self.dataframe[col], self.dataframe[self.variable_respuesta])
            return self.dataframe, encoder

        print("No se encontraron columnas para CatBoost.")
        return self.dataframe, None
    
def outliers_isolation_forest(df, niveles_contaminacion, lista_estimadores):
    """
    Agrega columnas al DataFrame con la detección de outliers utilizando Isolation Forest
    para diferentes niveles de contaminación y números de estimadores.

    Parámetros:
    - df (pd.DataFrame): El DataFrame de entrada.
    - niveles_contaminacion (list): Lista de niveles de contaminación a probar (por ejemplo, [0.01, 0.05, 0.1]).
    - lista_estimadores (list): Lista de cantidades de estimadores a probar (por ejemplo, [10, 100, 200]).

    Returns:
    - pd.DataFrame: DataFrame con nuevas columnas para cada configuración de Isolation Forest.
    """
    # Seleccionar columnas numéricas
    col_numericas = df.select_dtypes(include=np.number).columns

    # Generar todas las combinaciones de niveles de contaminación y estimadores
    combinaciones = list(product(niveles_contaminacion, lista_estimadores))

    for cont, esti in combinaciones:
        # Inicializar Isolation Forest
        ifo = IsolationForest(
            n_estimators=esti,
            contamination=cont,
            n_jobs=-1  # Usar todos los núcleos disponibles
        )

        # Ajustar y predecir outliers
        df[f"outliers_ifo_{cont}_{esti}"] = ifo.fit_predict(df[col_numericas])
    
    return df


def separar_dataframe(dataframe):
    """
    Separa un DataFrame en dos DataFrames: uno con columnas numéricas y otro con columnas categóricas.

    Parámetros:
    - dataframe (pd.DataFrame): El DataFrame original.

    Returns:
    - (pd.DataFrame, pd.DataFrame): Un DataFrame con columnas numéricas y otro con columnas categóricas.
    """
    return dataframe.select_dtypes(include=np.number), dataframe.select_dtypes(include="O")


def detectar_outliers(dataframe, color="orange", tamano_grafica=(15, 10)):
    """
    Muestra gráficos de caja para detectar outliers en columnas numéricas.

    Parámetros:
    - dataframe (pd.DataFrame): El DataFrame que contiene los datos.
    - color (str): Color para los gráficos.
    - tamano_grafica (tuple): Tamaño de la figura.

    Returns:
    - None: Muestra los gráficos directamente.
    """
    df_num = separar_dataframe(dataframe)[0]
    num_filas = math.ceil(len(df_num.columns) / 2)

    fig, axes = plt.subplots(ncols=2, nrows=num_filas, figsize=tamano_grafica)
    axes = axes.flat

    for indice, columna in enumerate(df_num.columns):
        sns.boxplot(
            x=columna,
            data=df_num,
            ax=axes[indice],
            color=color,
            flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 5}
        )
        axes[indice].set_title(f"Outliers de {columna}")
        axes[indice].set_xlabel("")

    if len(df_num.columns) % 2 != 0:
        fig.delaxes(axes[-1])
    plt.tight_layout()
    plt.show()


def aplicar_escaladores(df, columnas, escaladores, return_scaler=False):
    """
    Aplica múltiples escaladores a columnas específicas de un DataFrame y devuelve el DataFrame modificado.
    Puede opcionalmente devolver el primer escalador utilizado.

    Parámetros:
    - df: DataFrame de entrada en formato pandas.
    - columnas: Una lista de nombres de columnas que se quieren escalar.
    - escaladores: Una lista de instancias de escaladores (por ejemplo, [RobustScaler(), MinMaxScaler()]).
    - return_scaler: Booleano, si es True devuelve también el primer escalador utilizado.

    Retorna:
    - df_escalado: DataFrame con las columnas escaladas añadidas, con nombres de columnas que incluyen 
                   el sufijo correspondiente al nombre del escalador.
    - (Opcional) primer_escalador: El primer escalador utilizado para la transformación, si return_scaler=True.
    """
    # Crear una copia del DataFrame para no modificar el original
    df_escalado = df.copy()

    # Variable para guardar el primer escalador
    primer_escalador = None

    for i, escalador in enumerate(escaladores):
        # Ajustar y transformar las columnas seleccionadas
        datos_escalados = escalador.fit_transform(df[columnas])
        
        # Generar nombres de columnas basados en el nombre del escalador
        nombre_escalador = escalador.__class__.__name__.replace("Scaler", "").lower()
        nuevas_columnas = [f"{col}_{nombre_escalador}" for col in columnas]
        
        # Añadir las columnas escaladas al DataFrame
        df_escalado[nuevas_columnas] = datos_escalados

        # Guardar el primer escalador utilizado
        if i == 0:
            primer_escalador = escalador

    if return_scaler:
        return df_escalado, primer_escalador
    else:
        return df_escalado


