#!/usr/bin/env python3
"""
PySpark Streaming Job for Credit Card Fraud Detection

This script implements a PySpark Structured Streaming job that consumes transaction data from Kafka,
processes it for fraud detection, and outputs results to various destinations.
"""

import os
import sys
import json
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, udf, expr, window
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType, TimestampType, MapType

# Define schema for transaction data
transaction_schema = StructType([
    StructField("transaction_id", StringType(), True),
    StructField("timestamp", TimestampType(), True),
    StructField("card_id", StringType(), True),
    StructField("user_id", StringType(), True),
    StructField("merchant_id", StringType(), True),
    StructField("location_id", StringType(), True),
    StructField("device_id", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("currency", StringType(), True),
    StructField("status", StringType(), True),
    StructField("transaction_type", StringType(), True),
    StructField("payment_method", StringType(), True),
    StructField("is_online", BooleanType(), True),
    StructField("is_fraudulent", BooleanType(), True),
    StructField("fraud_score", DoubleType(), True),
    StructField("mcc", StringType(), True),
    StructField("auth_code", StringType(), True),
    StructField("response_code", StringType(), True),
    StructField("is_declined", BooleanType(), True),
    StructField("decline_reason", StringType(), True),
    StructField("user_profile", MapType(StringType(), StringType()), True),
    StructField("merchant_profile", MapType(StringType(), StringType()), True),
    StructField("location_context", MapType(StringType(), StringType()), True),
    StructField("device_context", MapType(StringType(), StringType()), True)
])

def create_spark_session(app_name="CreditCardFraudDetection"):
    """Create and configure Spark session"""
    return (SparkSession.builder
            .appName(app_name)
            .config("spark.sql.streaming.checkpointLocation", "/tmp/checkpoint")
            .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0")
            .getOrCreate())

def read_from_kafka(spark, bootstrap_servers, topic):
    """Read data from Kafka topic"""
    return (spark
            .readStream
            .format("kafka")
            .option("kafka.bootstrap.servers", bootstrap_servers)
            .option("subscribe", topic)
            .option("startingOffsets", "earliest")
            .load()
            .selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)", "timestamp")
            .select(
                col("key").alias("transaction_id"),
                from_json(col("value"), transaction_schema).alias("data"),
                col("timestamp").alias("kafka_timestamp")
            )
            .select("transaction_id", "data.*", "kafka_timestamp"))

def detect_fraud_patterns(df):
    """Apply fraud detection rules to the transaction data"""
    # Rule 1: High amount transactions
    high_amount_df = df.filter(col("amount") > 1000)
    
    # Rule 2: Transactions from high-risk merchants
    high_risk_merchant_df = df.filter(col("merchant_profile.is_high_risk") == "true")
    
    # Rule 3: Transactions with high fraud scores
    high_score_df = df.filter(col("fraud_score") > 0.7)
    
    # Rule 4: Transactions from suspicious devices
    suspicious_device_df = df.filter(
        (col("device_context.ip_is_proxy") == "true") | 
        (col("device_context.ip_is_vpn") == "true") | 
        (col("device_context.ip_is_tor") == "true")
    )
    
    # Rule 5: Transactions from users with fraud history
    fraud_history_df = df.filter(col("user_profile.fraud_history") == "true")
    
    # Combine all suspicious transactions
    suspicious_df = (high_amount_df.union(high_risk_merchant_df)
                    .union(high_score_df)
                    .union(suspicious_device_df)
                    .union(fraud_history_df)
                    .distinct())
    
    return suspicious_df

def analyze_transaction_patterns(df):
    """Analyze transaction patterns over time windows"""
    # Group by user_id and time window
    return (df.withWatermark("timestamp", "10 minutes")
            .groupBy(
                col("user_id"),
                window(col("timestamp"), "1 hour", "15 minutes")
            )
            .agg(
                expr("count(transaction_id)").alias("transaction_count"),
                expr("sum(amount)").alias("total_amount"),
                expr("avg(amount)").alias("avg_amount"),
                expr("count(distinct merchant_id)").alias("merchant_count"),
                expr("count(distinct location_id)").alias("location_count")
            )
            # Detect velocity abuse
            .filter(col("transaction_count") > 5)
            )

def process_transactions(spark, bootstrap_servers, input_topic, output_topic):
    """Main processing function for transaction data"""
    # Read from Kafka
    transactions_df = read_from_kafka(spark, bootstrap_servers, input_topic)
    
    # Apply fraud detection
    suspicious_df = detect_fraud_patterns(transactions_df)
    
    # Analyze patterns
    patterns_df = analyze_transaction_patterns(transactions_df)
    
    # Write suspicious transactions to Kafka
    suspicious_query = (suspicious_df
                        .selectExpr("transaction_id AS key", "to_json(struct(*)) AS value")
                        .writeStream
                        .format("kafka")
                        .option("kafka.bootstrap.servers", bootstrap_servers)
                        .option("topic", output_topic)
                        .option("checkpointLocation", "/tmp/checkpoint/suspicious")
                        .start())
    
    # Write pattern analysis to console (for demo purposes)
    patterns_query = (patterns_df
                     .writeStream
                     .outputMode("update")
                     .format("console")
                     .option("truncate", "false")
                     .start())
    
    # Write all transactions to console (for demo purposes)
    all_query = (transactions_df
                .writeStream
                .outputMode("append")
                .format("console")
                .option("truncate", "false")
                .start())
    
    # Wait for termination
    spark.streams.awaitAnyTermination()

def main():
    parser = argparse.ArgumentParser(description='PySpark Streaming for Credit Card Fraud Detection')
    parser.add_argument('--bootstrap-servers', type=str, default='localhost:9092',
                        help='Kafka bootstrap servers')
    parser.add_argument('--input-topic', type=str, default='enriched-transactions',
                        help='Kafka topic to consume from')
    parser.add_argument('--output-topic', type=str, default='fraud-alerts',
                        help='Kafka topic to produce to')
    
    args = parser.parse_args()
    
    spark = create_spark_session()
    
    try:
        process_transactions(
            spark,
            args.bootstrap_servers,
            args.input_topic,
            args.output_topic
        )
    except KeyboardInterrupt:
        print("Processing interrupted")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
