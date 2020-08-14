"""
Converts the raw CSV form to a Parquet form with just the columns we want
"""
import tempfile
import mlflow
import argparse

import os
import sys

os.environ['SPARK_HOME'] = "/databricks/spark/"
sys.path.append("/databricks/spark/python/")

from pyspark import SparkConf, SparkContext
from pyspark.sql.functions import *
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
import findspark





# file:///Users/logan.rudd/Work/repos/mlflow-mswf-poc/mlruns/0/297ca0ec1f634e3ea2d7f3631d76b310/artifacts/loans-raw-csv-dir
def etl_data(loans_csv_uri, spark):
    with mlflow.start_run(nested=True) as mlrun:
        tmpdir = tempfile.mkdtemp()
        loans_parquet_dir = os.path.join(tmpdir, 'loans-parquet')
        print("Converting ratings CSV %s to Parquet %s" % (loans_csv_uri,
                                                           loans_parquet_dir))

        loans = spark.read.option("header", "true")\
            .option("inferSchema", "true").csv(loans_csv_uri)

        print("Create bad loan label, this will include charged off, defaulted,"
              "and late repayments on loans...")
        loans = loans.filter(
            loans.loan_status.isin(["Default", "Charged Off", "Fully Paid"]))\
            .withColumn("bad_loan",
                        (~(loans.loan_status == "Fully Paid")).cast("int"))

        print("Casting numeric columns into the appropriate types...")
        loans = loans.withColumn('issue_year',
                                 substring(loans.issue_d, 5, 4).cast('double'))\
            .withColumn('earliest_year',
                        substring(loans.earliest_cr_line, 5, 4).cast('double'))
        loans = loans.withColumn('credit_length_in_years',
                                 (loans.issue_year - loans.earliest_year))

        for column in loans.columns:
            loans = loans.withColumn(column, loans[column].cast('double'))

        loans.write.parquet(loans_parquet_dir)
        print("Uploading Parquet loans: %s" % loans_parquet_dir)
        mlflow.log_artifacts(loans_parquet_dir, "loans-processed-parquet-dir")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--loans_csv_uri")
    args = parser.parse_args()
    findspark.init()
    spark_builder = SparkSession.builder.master("local[*]").appName("test")
    spark = spark_builder.getOrCreate()

    etl_data(args.loans_csv_uri, spark)
