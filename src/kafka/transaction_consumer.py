#!/usr/bin/env python3
"""
Kafka Consumer for Credit Card Fraud Detection

This script consumes transaction data from Kafka topics and processes it for fraud detection.
It demonstrates the integration between Kafka and PySpark for real-time processing.
"""

import json
import argparse
from kafka import KafkaConsumer
from kafka.errors import KafkaError

# Kafka topics
RAW_TRANSACTIONS_TOPIC = "raw-transactions"
ENRICHED_TRANSACTIONS_TOPIC = "enriched-transactions"
FRAUD_ALERTS_TOPIC = "fraud-alerts"
TRANSACTION_DISPOSITIONS_TOPIC = "transaction-dispositions"

class TransactionConsumer:
    def __init__(self, bootstrap_servers='localhost:9092', topic=ENRICHED_TRANSACTIONS_TOPIC, group_id='fraud-detection-group'):
        """Initialize Kafka consumer"""
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            auto_offset_reset='earliest',
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            key_deserializer=lambda k: k.decode('utf-8') if k else None
        )
        
        print(f"Consumer initialized for topic: {topic}")
    
    def consume_transactions(self, max_messages=None):
        """Consume transactions from Kafka topic"""
        print(f"Starting to consume transactions...")
        
        count = 0
        for message in self.consumer:
            transaction = message.value
            
            print(f"Received transaction {transaction['transaction_id']} from partition {message.partition}, offset {message.offset}")
            print(f"Transaction details: Amount=${transaction['amount']}, Merchant={transaction.get('merchant_profile', {}).get('category', 'Unknown')}")
            
            # Process transaction (in a real system, this would be more complex)
            if 'is_fraudulent' in transaction and transaction['is_fraudulent']:
                print(f"FRAUD ALERT: Transaction {transaction['transaction_id']} is fraudulent!")
                if 'fraud_pattern' in transaction:
                    print(f"Fraud pattern: {transaction['fraud_pattern']}")
            
            count += 1
            if max_messages and count >= max_messages:
                break
        
        print(f"Consumed {count} transactions")
    
    def close(self):
        """Close the consumer"""
        self.consumer.close()

def main():
    parser = argparse.ArgumentParser(description='Consume credit card transactions from Kafka')
    parser.add_argument('--bootstrap-servers', type=str, default='localhost:9092',
                        help='Kafka bootstrap servers')
    parser.add_argument('--topic', type=str, default=ENRICHED_TRANSACTIONS_TOPIC,
                        help='Kafka topic to consume from')
    parser.add_argument('--group-id', type=str, default='fraud-detection-group',
                        help='Consumer group ID')
    parser.add_argument('--max-messages', type=int, default=None,
                        help='Maximum number of messages to consume (default: unlimited)')
    
    args = parser.parse_args()
    
    consumer = TransactionConsumer(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        group_id=args.group_id
    )
    
    try:
        consumer.consume_transactions(args.max_messages)
    except KeyboardInterrupt:
        print("Consumption interrupted")
    finally:
        consumer.close()

if __name__ == "__main__":
    main()
