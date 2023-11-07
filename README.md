# En proceso


# Introdución
---

![Alt text](pics\image-1.png)


#### El fraude financiero representa un desafío significativo para bancos e instituciones financieras en todo el mundo, ya que puede resultar en pérdidas financieras sustanciales y dañar la reputación de una empresa. En los últimos años, el Aprendizaje Automático se ha convertido en una herramienta valiosa para mejorar la detección de fraudes financieros. En este proyecto, asumiré el rol de un científico de datos y desarrollaré un modelo capaz de detectar fraudes con un alto nivel de precisión.

---


# Dataset Seleccionado
---

Para llevar a cabo el siguiente proyecto seleccione el dataset publico

> *The dataset contains transactions made by credit cards in September 2013 by European cardholders.*
> 
> *This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.*
> 
> *It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.*
> 
> *Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.*

Observamos que el dataset ha sido preprocesado para mantener su anonimato mediante la técnica de reducción de dimensionalidad PCA. También notamos un gran desequilibrio entre las etiquetas positivas y negativas.

# Objetivos
---

- Evaluar el rendimiento de diversos algoritmos y realizar ajustes de hiperparámetros.

- Implementar técnicas para abordar el desequilibrio en el dataset.

- Alcanzar altos niveles de **AUC-ROC** ya que  es crucial utilizarlas en un dataset altamente desbalanceado. Las métricas "tradicionales" pueden dar resultados contraintuitivos. Por ejemplo, si clasificáramos todos los registros como "no fraudulentos", obtendríamos un Accuracy Score de más del 99%. La métrica AUC-ROC se convierte en una alternativa sólida en esta circunstancia.

- Seleccionar los modelos más efectivos y crear un conjunto de modelos para optimizar los resultados.

- Probar el conjunto de modelos con diferentes Meta-Clasificadores  

- LLegar a una conclusion de la rentabilidad del ensamblado de modelos.


###### [Articulo sobre meta clasificadores en modelos de prevencion de fraude financiero](https://www.sciencedirect.com/science/article/abs/pii/S1544612322001866).
---


# Tecnologías utilizadas

- ### Selección de Características::
    - Dado que el conjunto de datos ya ha sido preprocesado mediante un algoritmo de PCA, llevé a cabo la selección de características basándome en la importancia de las características del algoritmo **Random Forest Classifier.**

- ### Re-Balanceo del Dataset:
    - Realicé pruebas con tres técnicas diferentes: **UnderSampling, OverSampling y SMOTE** a través de la biblioteca imblearn. De las tres, la que proporcionó mejores resultados fue SMOTE, que genera datos sintéticos para la clase minoritaria al combinar instancias existentes con sus vecinos cercanos, abordando así el desequilibrio de clases en conjuntos de datos

- ### Selección de Algoritmos:
    - Inicié con 7 algoritmos diferentes:
        - **AdaBoost** (Adaptive Boosting): un algoritmo de aprendizaje automático que mejora el rendimiento de modelos de clasificación al dar más importancia a las instancias clasificadas incorrectamente en cada iteración, permitiendo que modelos débiles se combinen en un modelo fuerte para tomar decisiones más precisas.

        - **RUSBoost**: un algoritmo de aprendizaje automático que combina Random Under-Sampling (RUS) con AdaBoost para abordar el desequilibrio de clases en conjuntos de datos. RUS se utiliza para reducir la cantidad de ejemplos de la clase mayoritaria, y luego se aplica AdaBoost a los datos equilibrados para mejorar la clasificación.

        - **Random Forest**: un algoritmo de aprendizaje automático que construye un conjunto de árboles de decisión y utiliza su combinación para realizar predicciones más precisas y robustas.

        - **Extra Trees**: se toman múltiples muestras aleatorias de las características y se eligen los umbrales de manera más aleatoria en cada nodo del árbol de decisión, lo que lo hace aún más robusto y menos propenso al sobreajuste en comparación con Random Forest.

        - **KNN** (K-Nearest Neighbors): un algoritmo de aprendizaje supervisado que clasifica un punto de datos según la mayoría de sus k vecinos más cercanos en un espacio multidimensional.

        - **SVM** (Support Vector Machine): un algoritmo de aprendizaje automático que se utiliza para clasificar datos o realizar regresión. Busca encontrar un hiperplano de separación óptimo que maximice la distancia entre las clases en un espacio multidimensional.

        - **XGBOOST**: un algoritmo de aprendizaje automático que utiliza árboles de decisión y técnicas de aumento para mejorar la precisión de la clasificación y la regresión.

        Después de varias pruebas, se descartaron SVM, AdaBoost y RUSBoost.

- ### Model Ensambling

    - Realicé pruebas para evaluar la capacidad de modelos ensamblados. En estas pruebas se utilizaron 3 distintos tipos de ensambles: **Stackin, Soft Voting y Hard Voting**



    ##### [Articulo sobre stacking](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/)
    ##### [Articuo sobre Voting](https://ilyasbinsalih.medium.com/what-is-hard-and-soft-voting-in-machine-learning-2652676b6a32)
---



