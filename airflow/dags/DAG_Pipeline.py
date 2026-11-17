from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta

# import hàm từ ingestion.py
from ingestion import process_survey, process_purchase_to_kafka

PACKAGES = (
    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,"
    "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.10.0,"
    "org.projectnessie.nessie-integrations:nessie-spark-extensions-3.5_2.12:0.105.7,"
    "org.apache.hadoop:hadoop-aws:3.3.4"
)

with DAG(
    dag_id="lakehouse_pipeline",
    start_date=datetime(2026, 2, 23),
    schedule_interval=None,
    catchup=False,
    default_args={
        "owner": "airflow",
        "retries": 1,
        "retry_delay": timedelta(minutes=2)
    },
    tags=["lakehouse"]
) as dag:

    # ======================
    # INGESTION
    # ======================
    survey_bronze_task = PythonOperator(
        task_id="survey_to_bronze",
        python_callable=process_survey
    )

    purchase_task = PythonOperator(
        task_id="purchase_to_kafka",
        python_callable=process_purchase_to_kafka
    )

    # ======================
    # PURCHASE PIPELINE
    # ======================
    purchase_bronze_task = SparkSubmitOperator(
        task_id="run_bronze_stream",
        conn_id="spark_conn",
        application="/opt/airflow/dags/bronze_stream.py",
        conf={
            "spark.jars.packages": PACKAGES
        }
    )

    silver_purchase_task = SparkSubmitOperator(
        task_id="run_silver_stream",
        conn_id="spark_conn",
        application="/opt/airflow/dags/silver_stream.py",
        conf={
            "spark.jars.packages": PACKAGES
        }
    )

    # ======================
    # SURVEY PIPELINE
    # ======================
    survey_silver_task = SparkSubmitOperator(
        task_id="transform_survey_to_silver",
        conn_id="spark_conn",
        application="/opt/airflow/dags/transform_survey.py",
        application_args=["survey.csv"],
        conf={
            "spark.jars.packages": PACKAGES,
            "spark.sql.extensions":
                "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions,"
                "org.projectnessie.spark.extensions.NessieSparkSessionExtensions",
            "spark.sql.catalog.nessie":
                "org.apache.iceberg.spark.SparkCatalog",
            "spark.sql.catalog.nessie.uri":
                "http://nessie:19120/api/v1",
            "spark.sql.catalog.nessie.warehouse":
                "s3a://silver/",
            "spark.sql.catalog.nessie.catalog-impl":
                "org.apache.iceberg.nessie.NessieCatalog",
            "spark.sql.catalog.nessie.io-impl":
                "org.apache.iceberg.hadoop.HadoopFileIO",
            "spark.hadoop.fs.s3a.endpoint":
                "http://minio:9000",
            "spark.hadoop.fs.s3a.access.key":
                "minioadmin",
            "spark.hadoop.fs.s3a.secret.key":
                "minioadmin",
            "spark.hadoop.fs.s3a.path.style.access":
                "true",
            "spark.hadoop.fs.s3a.impl":
                "org.apache.hadoop.fs.s3a.S3AFileSystem"
        }
    )

    # ======================
    # GOLD
    # ======================
    gold_task = SparkSubmitOperator(
        task_id="process_gold",
        conn_id="spark_conn",
        application="/opt/airflow/dags/process_gold_layer.py",
        conf={
            "spark.jars.packages": PACKAGES,
            "spark.master": "spark://spark-master:7077",
            "spark.executor.memory": "4g",
            "spark.driver.memory": "2g"
        }
    )

    # ======================
    # DEPENDENCIES
    # ======================

    purchase_task >> purchase_bronze_task >> silver_purchase_task

    survey_bronze_task >> survey_silver_task

    [silver_purchase_task, survey_silver_task] >> gold_task