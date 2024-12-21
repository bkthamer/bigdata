from airflow import DAG

from airflow.operators.python import PythonOperator

from hdfs import InsecureClient, HdfsError

import requests

import pandas as pd

import datetime

from pyspark.sql import SparkSession

from pyspark.ml.feature import VectorAssembler, StringIndexer

from pyspark.ml.regression import LinearRegression


# Initialisation du client HDFS

hdfs_client = InsecureClient('http://localhost:9871', user='hadoop')


def fetch_data(**context):

    url = "https://restcountries.com/v3.1/all"

    response = requests.get(url)

    data = response.json()

    df = pd.json_normalize(data)

    local_path = '/tmp/countries_data.csv'

    df.to_csv(local_path, index=False)

    return local_path


def upload_to_hdfs(**context):

    local_path = context['task_instance'].xcom_pull(task_ids='fetch_data')

    hdfs_path = '/user/hadoop/countries_data3.csv'

    try:

        hdfs_client.upload(hdfs_path, local_path, overwrite=True)

        print(f"Fichier téléchargé avec succès sur HDFS à {hdfs_path}")

    except HdfsError as e:

        print(f"Erreur lors du téléchargement sur HDFS: {str(e)}")

        raise


def upload_kaggle_to_hdfs(**context):

    local_path = '/home/thamer/countrieskaggle.csv'

    hdfs_path = '/user/hadoop/countries_kaggle.csv'

    try:

        hdfs_client.upload(hdfs_path, local_path, overwrite=True)

        print(f"Fichier Kaggle téléchargé avec succès sur HDFS à {hdfs_path}")

    except HdfsError as e:

        print(f"Erreur lors du téléchargement du fichier Kaggle sur HDFS: {str(e)}")

        raise


def download_from_hdfs(**context):

    hdfs_path = '/user/hadoop/countries_data3.csv'

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    local_path = f'/home/thamer/countries_data_{timestamp}.csv'

    try:

        hdfs_client.download(hdfs_path, local_path)

        print(f"Fichier téléchargé depuis HDFS à {local_path}")

        context['task_instance'].xcom_push(key='local_path', value=local_path)

    except HdfsError as e:

        print(f"Erreur lors du téléchargement depuis HDFS: {str(e)}")

        raise


def display_csv(**context):

    local_path = context['task_instance'].xcom_pull(task_ids='download_from_hdfs', key='local_path')

    df = pd.read_csv(local_path)

    print(df)


def reduce_columns(**context):

    local_path = context['task_instance'].xcom_pull(task_ids='fetch_data')

    df = pd.read_csv(local_path)

    df_reduced = df[['cca2', 'cca3', 'ccn3', 'cioc', 'population']]

    reduced_path = '/home/thamer/csvreduit.csv'

    df_reduced.to_csv(reduced_path, index=False)

    print(f"Fichier réduit créé avec succès à {reduced_path}")


def merge_csv_files(**context):

    reduced_path = '/home/thamer/csvreduit.csv'

    kaggle_path = '/home/thamer/countrieskaggle.csv'

    merged_path = '/home/thamer/csvdouble.csv'

    df_reduced = pd.read_csv(reduced_path)

    df_kaggle = pd.read_csv(kaggle_path, usecols=['Country', 'Region'])

    df_merged = pd.concat([df_reduced.reset_index(drop=True), df_kaggle.reset_index(drop=True)], axis=1)

    df_merged.to_csv(merged_path, index=False)

    print(f"Fichier fusionné avec nouvelles colonnes créé avec succès à {merged_path}")


def train_model(**context):

    # Initialiser la session Spark

    spark = SparkSession.builder.appName("LinearRegressionExample").getOrCreate()

    

    # Charger le fichier csvdouble.csv

    data = spark.read.csv("/home/thamer/csvdouble.csv", header=True, inferSchema=True)

    

    # Convertir les colonnes de chaînes de caractères en numériques avec StringIndexer

    indexers = [

        StringIndexer(inputCol=col, outputCol=col + "_indexed", handleInvalid="skip").fit(data)

        for col in ["cca2", "cca3", "cioc"]

    ]

    

    for indexer in indexers:

        data = indexer.transform(data)

    

    # Remplacer les valeurs NULL par 0 dans les colonnes indexées

    data = data.na.fill({"cca2_indexed": 0, "cca3_indexed": 0, "ccn3": 0, "cioc_indexed": 0})

    

    # Sélectionner les colonnes indexées pour les assembler en une seule colonne de caractéristiques

    feature_columns = ["cca2_indexed", "cca3_indexed", "ccn3", "cioc_indexed"]

    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

    prepared_data = assembler.transform(data)

    

    # Sélectionner les colonnes 'features' et 'population' (cible)

    final_data = prepared_data.select("features", "population")

    

    # Diviser les données en ensemble d'entraînement (70%) et de test (30%)

    train_data, test_data = final_data.randomSplit([0.7, 0.3])

    

    # Initialiser le modèle de régression linéaire

    lr = LinearRegression(labelCol="population")

    

    # Entraîner le modèle

    lr_model = lr.fit(train_data)

    

    # Afficher les coefficients et l'intercept

    print(f"Coefficients: {lr_model.coefficients}")

    print(f"Intercept: {lr_model.intercept}")

    

    # Évaluer le modèle sur les données de test

    test_results = lr_model.evaluate(test_data)

    

    # Afficher les métriques

    print(f"RMSE: {test_results.rootMeanSquaredError}")

    print(f"R2: {test_results.r2}")


# Configuration par défaut

default_args = {

    'owner': 'airflow',

    'start_date': datetime.datetime(2024, 10, 19),

}


with DAG('fetch_countries_data', default_args=default_args, schedule_interval=None) as dag:

    fetch_data_task = PythonOperator(

        task_id='fetch_data',

        python_callable=fetch_data,

        provide_context=True,

    )


    upload_data_task = PythonOperator(

        task_id='upload_to_hdfs',

        python_callable=upload_to_hdfs,

        provide_context=True,

    )


    upload_kaggle_task = PythonOperator(

        task_id='upload_kaggle_to_hdfs',

        python_callable=upload_kaggle_to_hdfs,

        provide_context=True,

    )


    download_data_task = PythonOperator(

        task_id='download_from_hdfs',

        python_callable=download_from_hdfs,

        provide_context=True,

    )


    display_data_task = PythonOperator(

        task_id='display_csv',

        python_callable=display_csv,

        provide_context=True,

    )


    reduce_columns_task = PythonOperator(

        task_id='reduce_columns',

        python_callable=reduce_columns,

        provide_context=True,

    )


    merge_csv_task = PythonOperator(

        task_id='merge_csv_files',

        python_callable=merge_csv_files,

        provide_context=True,

    )


    train_model_task = PythonOperator(

        task_id='train_model',

        python_callable=train_model,

        provide_context=True,

    )


    # Définition des dépendances des tâches

    fetch_data_task >> upload_data_task >> upload_kaggle_task >> download_data_task >> reduce_columns_task >> merge_csv_task >> train_model_task >> display_data_task



#thamer
