#!/usr/bin/env python3
"""
Kafka Producer for Credit Card Fraud Detection

This script reads transaction data from CSV files and publishes it to Kafka topics.
It simulates real-time transaction streaming by publishing transactions with timestamps.
"""

import json
import time
import pandas as pd
import argparse
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Kafka topics
RAW_TRANSACTIONS_TOPIC = "raw-transactions"
ENRICHED_TRANSACTIONS_TOPIC = "enriched-transactions"
FRAUD_ALERTS_TOPIC = "fraud-alerts"
TRANSACTION_DISPOSITIONS_TOPIC = "transaction-dispositions"

class TransactionProducer:
    def __init__(self, bootstrap_servers='localhost:9092'):
        """Initialize Kafka producer"""
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
        
        # Load reference data
        self.users = pd.read_csv('/home/ubuntu/credit_card_fraud_detection/data/users.csv')
        self.cards = pd.read_csv('/home/ubuntu/credit_card_fraud_detection/data/cards.csv')
        self.merchants = pd.read_csv('/home/ubuntu/credit_card_fraud_detection/data/merchants.csv')
        self.locations = pd.read_csv('/home/ubuntu/credit_card_fraud_detection/data/locations.csv')
        self.devices = pd.read_csv('/home/ubuntu/credit_card_fraud_detection/data/devices.csv')
        self.ip_addresses = pd.read_csv('/home/ubuntu/credit_card_fraud_detection/data/ip_addresses.csv')
        
        # Convert to dictionaries for faster lookup
        self.users_dict = self.users.set_index('user_id').to_dict(orient='index')
        self.cards_dict = self.cards.set_index('card_id').to_dict(orient='index')
        self.merchants_dict = self.merchants.set_index('merchant_id').to_dict(orient='index')
        self.locations_dict = self.locations.set_index('location_id').to_dict(orient='index')
        self.devices_dict = self.devices.set_index('device_id').to_dict(orient='index')
        self.ip_dict = self.ip_addresses.set_index('ip').to_dict(orient='index')
        
        print(f"Loaded reference data: {len(self.users)} users, {len(self.cards)} cards, "
              f"{len(self.merchants)} merchants, {len(self.locations)} locations, "
              f"{len(self.devices)} devices")
    
    def publish_transaction(self, transaction, topic=RAW_TRANSACTIONS_TOPIC):
        """Publish a transaction to Kafka topic"""
        # Use transaction_id as key for partitioning
        future = self.producer.send(
            topic,
            key=transaction['transaction_id'],
            value=transaction
        )
        
        # Handle success/failure
        try:
            record_metadata = future.get(timeout=10)
            print(f"Published transaction {transaction['transaction_id']} to {record_metadata.topic} "
                  f"partition {record_metadata.partition} offset {record_metadata.offset}")
            return True
        except KafkaError as e:
            print(f"Failed to publish transaction {transaction['transaction_id']}: {e}")
            return False
    
    def enrich_transaction(self, transaction):
        """Enrich transaction with additional context from reference data"""
        # Get user data
        user_id = transaction.get('user_id')
        user_data = self.users_dict.get(user_id, {})
        
        # Get card data
        card_id = transaction.get('card_id')
        card_data = self.cards_dict.get(card_id, {})
        
        # Get merchant data
        merchant_id = transaction.get('merchant_id')
        merchant_data = self.merchants_dict.get(merchant_id, {})
        
        # Get location data
        location_id = transaction.get('location_id')
        location_data = self.locations_dict.get(location_id, {})
        
        # Get device and IP data for online transactions
        device_data = {}
        ip_data = {}
        if transaction.get('is_online'):
            device_id = transaction.get('device_id')
            if device_id:
                device_data = self.devices_dict.get(device_id, {})
                ip = device_data.get('ip')
                if ip:
                    ip_data = self.ip_dict.get(ip, {})
        
        # Create enriched transaction
        enriched_transaction = transaction.copy()
        
        # Add user profile
        enriched_transaction['user_profile'] = {
            'risk_score': user_data.get('risk_score'),
            'account_age_days': (datetime.now() - datetime.fromisoformat(user_data.get('account_creation_date', '2020-01-01'))).days,
            'typical_transaction_amount': user_data.get('typical_transaction_amount'),
            'fraud_history': user_data.get('fraud_history'),
            'segment': user_data.get('segment')
        }
        
        # Add merchant profile
        enriched_transaction['merchant_profile'] = {
            'category': merchant_data.get('category'),
            'risk_score': merchant_data.get('risk_score'),
            'fraud_rate': merchant_data.get('fraud_rate'),
            'avg_transaction_amount': merchant_data.get('avg_transaction_amount'),
            'is_high_risk': merchant_data.get('is_high_risk')
        }
        
        # Add location context
        enriched_transaction['location_context'] = {
            'country': location_data.get('country'),
            'city': location_data.get('city'),
            'risk_score': location_data.get('risk_score')
        }
        
        # Add device and IP context for online transactions
        if transaction.get('is_online'):
            enriched_transaction['device_context'] = {
                'device_type': device_data.get('device_type'),
                'is_mobile': device_data.get('is_mobile'),
                'is_known_device': device_data.get('is_known_device'),
                'risk_score': device_data.get('risk_score'),
                'ip': device_data.get('ip'),
                'ip_country': ip_data.get('country'),
                'ip_is_proxy': ip_data.get('is_proxy'),
                'ip_is_vpn': ip_data.get('is_vpn'),
                'ip_is_tor': ip_data.get('is_tor'),
                'ip_risk_score': ip_data.get('risk_score')
            }
        
        return enriched_transaction
    
    def stream_transactions(self, transactions_file, rate=10, enrich=True):
        """Stream transactions from file to Kafka at specified rate"""
        # Load transactions
        print(f"Loading transactions from {transactions_file}")
        transactions_df = pd.read_csv(transactions_file)
        transactions = transactions_df.to_dict(orient='records')
        
        print(f"Streaming {len(transactions)} transactions at rate of {rate} per second")
        
        # Stream transactions
        for i, transaction in enumerate(transactions):
            # Convert transaction to proper format
            tx = {k: v for k, v in transaction.items() if pd.notna(v)}
            
            # Publish raw transaction
            success = self.publish_transaction(tx, RAW_TRANSACTIONS_TOPIC)
            
            if success and enrich:
                # Enrich and publish enriched transaction
                enriched_tx = self.enrich_transaction(tx)
                self.publish_transaction(enriched_tx, ENRICHED_TRANSACTIONS_TOPIC)
                
                # If transaction is fraudulent, publish to fraud alerts topic
                if tx.get('is_fraudulent'):
                    self.publish_transaction(enriched_tx, FRAUD_ALERTS_TOPIC)
            
            # Sleep to control rate
            if i < len(transactions) - 1:  # Don't sleep after last transaction
                time.sleep(1.0 / rate)
            
            # Print progress
            if (i + 1) % 100 == 0:
                print(f"Published {i + 1}/{len(transactions)} transactions")
        
        print(f"Finished streaming {len(transactions)} transactions")
    
    def close(self):
        """Close the producer"""
        self.producer.flush()
        self.producer.close()

def main():
    parser = argparse.ArgumentParser(description='Stream credit card transactions to Kafka')
    parser.add_argument('--file', type=str, default='/home/ubuntu/credit_card_fraud_detection/data/all_transactions.csv',
                        help='Path to transactions CSV file')
    parser.add_argument('--bootstrap-servers', type=str, default='localhost:9092',
                        help='Kafka bootstrap servers')
    parser.add_argument('--rate', type=int, default=10,
                        help='Transactions per second')
    parser.add_argument('--no-enrich', action='store_true',
                        help='Disable transaction enrichment')
    
    args = parser.parse_args()
    
    producer = TransactionProducer(bootstrap_servers=args.bootstrap_servers)
    
    try:
        producer.stream_transactions(args.file, args.rate, not args.no_enrich)
    except KeyboardInterrupt:
        print("Streaming interrupted")
    finally:
        producer.close()

if __name__ == "__main__":
    main()
