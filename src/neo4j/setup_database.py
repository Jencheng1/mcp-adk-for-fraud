#!/usr/bin/env python3
"""
Neo4j Database Setup for Credit Card Fraud Detection

This script sets up the Neo4j database with the schema designed for fraud detection.
It creates constraints, indexes, and loads the initial data from CSV files.
"""

import os
import json
import argparse
import pandas as pd
from neo4j import GraphDatabase

class Neo4jFraudDetection:
    def __init__(self, uri, username, password):
        """Initialize Neo4j connection"""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        print(f"Connected to Neo4j at {uri}")
    
    def close(self):
        """Close the Neo4j connection"""
        self.driver.close()
    
    def run_query(self, query, parameters=None):
        """Run a Cypher query"""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return list(result)
    
    def create_constraints_and_indexes(self):
        """Create constraints and indexes for the fraud detection schema"""
        print("Creating constraints and indexes...")
        
        # Unique constraints
        constraints = [
            "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.userId IS UNIQUE",
            "CREATE CONSTRAINT card_id_unique IF NOT EXISTS FOR (c:Card) REQUIRE c.cardId IS UNIQUE",
            "CREATE CONSTRAINT transaction_id_unique IF NOT EXISTS FOR (t:Transaction) REQUIRE t.transactionId IS UNIQUE",
            "CREATE CONSTRAINT merchant_id_unique IF NOT EXISTS FOR (m:Merchant) REQUIRE m.merchantId IS UNIQUE",
            "CREATE CONSTRAINT device_id_unique IF NOT EXISTS FOR (d:Device) REQUIRE d.deviceId IS UNIQUE",
            "CREATE CONSTRAINT ip_unique IF NOT EXISTS FOR (i:IPAddress) REQUIRE i.ip IS UNIQUE",
            "CREATE CONSTRAINT alert_id_unique IF NOT EXISTS FOR (a:Alert) REQUIRE a.alertId IS UNIQUE",
            "CREATE CONSTRAINT location_id_unique IF NOT EXISTS FOR (l:Location) REQUIRE l.locationId IS UNIQUE"
        ]
        
        # Indexes
        indexes = [
            "CREATE INDEX user_email_idx IF NOT EXISTS FOR (u:User) ON (u.email)",
            "CREATE INDEX card_last_four_idx IF NOT EXISTS FOR (c:Card) ON (c.lastFourDigits)",
            "CREATE INDEX transaction_timestamp_idx IF NOT EXISTS FOR (t:Transaction) ON (t.timestamp)",
            "CREATE INDEX merchant_name_idx IF NOT EXISTS FOR (m:Merchant) ON (m.name)",
            "CREATE INDEX merchant_category_idx IF NOT EXISTS FOR (m:Merchant) ON (m.category)",
            "CREATE INDEX location_country_idx IF NOT EXISTS FOR (l:Location) ON (l.country)",
            "CREATE INDEX device_fingerprint_idx IF NOT EXISTS FOR (d:Device) ON (d.fingerprint)",
            "CREATE INDEX ip_country_idx IF NOT EXISTS FOR (i:IPAddress) ON (i.country)"
        ]
        
        # Execute all constraints and indexes
        for constraint in constraints:
            self.run_query(constraint)
            print(f"Created constraint: {constraint}")
        
        for index in indexes:
            self.run_query(index)
            print(f"Created index: {index}")
    
    def load_users(self, users_file):
        """Load user data into Neo4j"""
        print(f"Loading users from {users_file}...")
        
        # Read CSV file
        users_df = pd.read_csv(users_file)
        
        # Create Cypher query for batch import
        query = """
        UNWIND $users AS user
        MERGE (u:User {userId: user.user_id})
        SET u.name = user.name,
            u.email = user.email,
            u.phone = user.phone,
            u.riskScore = toFloat(user.risk_score),
            u.accountCreationDate = datetime(user.account_creation_date),
            u.lastActivityDate = datetime(user.last_activity_date),
            u.fraudHistory = toBoolean(user.fraud_history),
            u.kycVerified = toBoolean(user.kyc_verified),
            u.segment = user.segment,
            u.typicalTransactionAmount = toFloat(user.typical_transaction_amount)
        RETURN count(u) AS userCount
        """
        
        # Execute in batches to avoid memory issues
        batch_size = 100
        total_users = 0
        
        for i in range(0, len(users_df), batch_size):
            batch = users_df.iloc[i:i+batch_size].to_dict(orient='records')
            result = self.run_query(query, {"users": batch})
            total_users += result[0]["userCount"]
            print(f"Loaded {total_users}/{len(users_df)} users")
    
    def load_cards(self, cards_file):
        """Load card data into Neo4j"""
        print(f"Loading cards from {cards_file}...")
        
        # Read CSV file
        cards_df = pd.read_csv(cards_file)
        
        # Create Cypher query for batch import
        query = """
        UNWIND $cards AS card
        MATCH (u:User {userId: card.user_id})
        MERGE (c:Card {cardId: card.card_id})
        SET c.cardType = card.card_type,
            c.issueDate = datetime(card.issue_date),
            c.expiryDate = datetime(card.expiry_date),
            c.lastFourDigits = card.last_four_digits,
            c.isActive = toBoolean(card.is_active),
            c.isBlocked = toBoolean(card.is_blocked),
            c.creditLimit = toFloat(card.credit_limit),
            c.availableCredit = toFloat(card.available_credit)
        MERGE (u)-[:OWNS]->(c)
        RETURN count(c) AS cardCount
        """
        
        # Execute in batches to avoid memory issues
        batch_size = 100
        total_cards = 0
        
        for i in range(0, len(cards_df), batch_size):
            batch = cards_df.iloc[i:i+batch_size].to_dict(orient='records')
            result = self.run_query(query, {"cards": batch})
            total_cards += result[0]["cardCount"]
            print(f"Loaded {total_cards}/{len(cards_df)} cards")
    
    def load_merchants(self, merchants_file):
        """Load merchant data into Neo4j"""
        print(f"Loading merchants from {merchants_file}...")
        
        # Read CSV file
        merchants_df = pd.read_csv(merchants_file)
        
        # Create Cypher query for batch import
        query = """
        UNWIND $merchants AS merchant
        MERGE (m:Merchant {merchantId: merchant.merchant_id})
        SET m.name = merchant.name,
            m.category = merchant.category,
            m.mcc = merchant.mcc,
            m.website = merchant.website,
            m.country = merchant.country,
            m.riskScore = toFloat(merchant.risk_score),
            m.fraudRate = toFloat(merchant.fraud_rate),
            m.avgTransactionAmount = toFloat(merchant.avg_transaction_amount),
            m.isHighRisk = toBoolean(merchant.is_high_risk)
        RETURN count(m) AS merchantCount
        """
        
        # Execute in batches to avoid memory issues
        batch_size = 100
        total_merchants = 0
        
        for i in range(0, len(merchants_df), batch_size):
            batch = merchants_df.iloc[i:i+batch_size].to_dict(orient='records')
            result = self.run_query(query, {"merchants": batch})
            total_merchants += result[0]["merchantCount"]
            print(f"Loaded {total_merchants}/{len(merchants_df)} merchants")
    
    def load_locations(self, locations_file):
        """Load location data into Neo4j"""
        print(f"Loading locations from {locations_file}...")
        
        # Read CSV file
        locations_df = pd.read_csv(locations_file)
        
        # Create Cypher query for batch import
        query = """
        UNWIND $locations AS location
        MERGE (l:Location {locationId: location.location_id})
        SET l.latitude = toFloat(location.latitude),
            l.longitude = toFloat(location.longitude),
            l.country = location.country,
            l.city = location.city,
            l.postalCode = location.postal_code,
            l.address = location.address,
            l.timezone = location.timezone,
            l.riskScore = toFloat(location.risk_score),
            l.point = point({latitude: toFloat(location.latitude), longitude: toFloat(location.longitude)})
        RETURN count(l) AS locationCount
        """
        
        # Execute in batches to avoid memory issues
        batch_size = 100
        total_locations = 0
        
        for i in range(0, len(locations_df), batch_size):
            batch = locations_df.iloc[i:i+batch_size].to_dict(orient='records')
            result = self.run_query(query, {"locations": batch})
            total_locations += result[0]["locationCount"]
            print(f"Loaded {total_locations}/{len(locations_df)} locations")
    
    def load_devices_and_ips(self, devices_file, ips_file):
        """Load device and IP data into Neo4j"""
        print(f"Loading devices from {devices_file} and IPs from {ips_file}...")
        
        # Read CSV files
        devices_df = pd.read_csv(devices_file)
        ips_df = pd.read_csv(ips_file)
        
        # Create IP addresses first
        ip_query = """
        UNWIND $ips AS ip_data
        MERGE (i:IPAddress {ip: ip_data.ip})
        SET i.country = ip_data.country,
            i.city = ip_data.city,
            i.isp = ip_data.isp,
            i.isProxy = toBoolean(ip_data.is_proxy),
            i.isVpn = toBoolean(ip_data.is_vpn),
            i.isTor = toBoolean(ip_data.is_tor),
            i.riskScore = toFloat(ip_data.risk_score)
        RETURN count(i) AS ipCount
        """
        
        # Execute IP import in batches
        batch_size = 100
        total_ips = 0
        
        for i in range(0, len(ips_df), batch_size):
            batch = ips_df.iloc[i:i+batch_size].to_dict(orient='records')
            result = self.run_query(ip_query, {"ips": batch})
            total_ips += result[0]["ipCount"]
            print(f"Loaded {total_ips}/{len(ips_df)} IP addresses")
        
        # Now create devices and link to IPs
        device_query = """
        UNWIND $devices AS device
        MATCH (ip:IPAddress {ip: device.ip})
        MERGE (d:Device {deviceId: device.device_id})
        SET d.deviceType = device.device_type,
            d.browser = device.browser,
            d.operatingSystem = device.operating_system,
            d.fingerprint = device.fingerprint,
            d.isMobile = toBoolean(device.is_mobile),
            d.isKnownDevice = toBoolean(device.is_known_device),
            d.riskScore = toFloat(device.risk_score)
        MERGE (d)-[:FROM]->(ip)
        RETURN count(d) AS deviceCount
        """
        
        # Execute device import in batches
        total_devices = 0
        
        for i in range(0, len(devices_df), batch_size):
            batch = devices_df.iloc[i:i+batch_size].to_dict(orient='records')
            result = self.run_query(device_query, {"devices": batch})
            total_devices += result[0]["deviceCount"]
            print(f"Loaded {total_devices}/{len(devices_df)} devices")
    
    def load_transactions(self, transactions_file, batch_size=50):
        """Load transaction data into Neo4j"""
        print(f"Loading transactions from {transactions_file}...")
        
        # Read CSV file
        transactions_df = pd.read_csv(transactions_file)
        
        # Create Cypher query for batch import
        query = """
        UNWIND $transactions AS tx
        MATCH (c:Card {cardId: tx.card_id})
        MATCH (m:Merchant {merchantId: tx.merchant_id})
        MATCH (l:Location {locationId: tx.location_id})
        OPTIONAL MATCH (d:Device {deviceId: tx.device_id})
        
        MERGE (t:Transaction {transactionId: tx.transaction_id})
        SET t.timestamp = datetime(tx.timestamp),
            t.amount = toFloat(tx.amount),
            t.currency = tx.currency,
            t.status = tx.status,
            t.transactionType = tx.transaction_type,
            t.paymentMethod = tx.payment_method,
            t.isOnline = toBoolean(tx.is_online),
            t.isFraudulent = toBoolean(tx.is_fraudulent),
            t.fraudScore = toFloat(tx.fraud_score),
            t.mcc = tx.mcc,
            t.authCode = tx.auth_code,
            t.responseCode = tx.response_code,
            t.isDeclined = toBoolean(tx.is_declined),
            t.declineReason = tx.decline_reason
        
        MERGE (c)-[:MADE]->(t)
        MERGE (t)-[:AT]->(m)
        MERGE (t)-[:IN]->(l)
        
        WITH t, d, tx
        WHERE d IS NOT NULL
        MERGE (t)-[:USING]->(d)
        
        RETURN count(t) AS txCount
        """
        
        # Execute in batches to avoid memory issues
        total_transactions = 0
        
        for i in range(0, len(transactions_df), batch_size):
            batch = transactions_df.iloc[i:i+batch_size].to_dict(orient='records')
            result = self.run_query(query, {"transactions": batch})
            total_transactions += result[0]["txCount"]
            print(f"Loaded {total_transactions}/{len(transactions_df)} transactions")
    
    def create_transaction_relationships(self):
        """Create relationships between transactions (FOLLOWED_BY)"""
        print("Creating relationships between transactions...")
        
        query = """
        MATCH (c:Card)-[:MADE]->(t1:Transaction)
        WITH c, t1
        ORDER BY t1.timestamp
        WITH c, collect(t1) as transactions
        UNWIND range(0, size(transactions)-2) as i
        WITH c, transactions[i] as t1, transactions[i+1] as t2
        WHERE duration.between(t1.timestamp, t2.timestamp).seconds < 86400  // Within 24 hours
        MERGE (t1)-[r:FOLLOWED_BY]->(t2)
        SET r.timeDifference = duration.between(t1.timestamp, t2.timestamp).seconds,
            r.velocityAnomaly = CASE 
                WHEN duration.between(t1.timestamp, t2.timestamp).seconds < 300 THEN true 
                ELSE false 
            END
        RETURN count(r) AS relationshipCount
        """
        
        result = self.run_query(query)
        print(f"Created {result[0]['relationshipCount']} FOLLOWED_BY relationships")
    
    def create_user_merchant_relationships(self):
        """Create relationships between users and frequently visited merchants"""
        print("Creating relationships between users and merchants...")
        
        query = """
        MATCH (u:User)-[:OWNS]->(c:Card)-[:MADE]->(t:Transaction)-[:AT]->(m:Merchant)
        WITH u, m, count(t) as frequency
        WHERE frequency >= 3
        MERGE (u)-[r:FREQUENTLY_VISITS]->(m)
        SET r.frequency = frequency,
            r.lastVisit = datetime()
        RETURN count(r) AS relationshipCount
        """
        
        result = self.run_query(query)
        print(f"Created {result[0]['relationshipCount']} FREQUENTLY_VISITS relationships")
    
    def create_user_location_relationships(self):
        """Create relationships between users and typical locations"""
        print("Creating relationships between users and locations...")
        
        query = """
        MATCH (u:User)-[:OWNS]->(c:Card)-[:MADE]->(t:Transaction)-[:IN]->(l:Location)
        WITH u, l, count(t) as frequency
        WHERE frequency >= 3
        MERGE (u)-[r:TYPICALLY_IN]->(l)
        SET r.frequency = frequency,
            r.lastVisit = datetime()
        RETURN count(r) AS relationshipCount
        """
        
        result = self.run_query(query)
        print(f"Created {result[0]['relationshipCount']} TYPICALLY_IN relationships")
    
    def create_fraud_patterns(self):
        """Create fraud pattern nodes and relationships to fraudulent transactions"""
        print("Creating fraud pattern nodes...")
        
        # First, create the fraud pattern nodes
        patterns_query = """
        UNWIND $patterns AS pattern
        MERGE (fp:FraudPattern {patternId: pattern.id})
        SET fp.description = pattern.description,
            fp.confidence = pattern.confidence,
            fp.discoveryDate = datetime(),
            fp.patternType = pattern.type,
            fp.isActive = true
        RETURN count(fp) AS patternCount
        """
        
        patterns = [
            {"id": "pattern_001", "type": "card_testing", "description": "Multiple small transactions in short time", "confidence": 0.9},
            {"id": "pattern_002", "type": "identity_theft", "description": "Unusual location and spending pattern", "confidence": 0.85},
            {"id": "pattern_003", "type": "account_takeover", "description": "Sudden change in behavior", "confidence": 0.8},
            {"id": "pattern_004", "type": "card_not_present", "description": "Online transactions with suspicious patterns", "confidence": 0.75},
            {"id": "pattern_005", "type": "merchant_fraud", "description": "Transactions at high-risk merchants", "confidence": 0.7},
            {"id": "pattern_006", "type": "application_fraud", "description": "New account with unusual activity", "confidence": 0.85},
            {"id": "pattern_007", "type": "transaction_laundering", "description": "Complex chain of transactions", "confidence": 0.8},
            {"id": "pattern_008", "type": "velocity_abuse", "description": "Many transactions in short time", "confidence": 0.9},
            {"id": "pattern_009", "type": "location_anomaly", "description": "Transactions in multiple distant locations", "confidence": 0.85},
            {"id": "pattern_010", "type": "amount_anomaly", "description": "Unusually large transaction amounts", "confidence": 0.8}
        ]
        
        result = self.run_query(patterns_query, {"patterns": patterns})
        print(f"Created {result[0]['patternCount']} fraud pattern nodes")
        
        # Now, connect fraudulent transactions to their patterns
        connect_query = """
        MATCH (t:Transaction {isFraudulent: true})
        WHERE t.fraudPattern IS NOT NULL
        MATCH (fp:FraudPattern)
        WHERE fp.patternType = t.fraudPattern
        MERGE (t)-[r:SIMILAR_TO]->(fp)
        SET r.similarityScore = 0.8 + rand() * 0.2,  // Random score between 0.8 and 1.0
            r.matchedFeatures = ["amount", "location", "timing"]
        RETURN count(r) AS relationshipCount
        """
        
        result = self.run_query(connect_query)
        print(f"Created {result[0]['relationshipCount']} SIMILAR_TO relationships")
    
    def setup_database(self, data_dir):
        """Set up the entire database with all data"""
        # Create constraints and indexes
        self.create_constraints_and_indexes()
        
        # Load all data
        self.load_users(os.path.join(data_dir, "users.csv"))
        self.load_cards(os.path.join(data_dir, "cards.csv"))
        self.load_merchants(os.path.join(data_dir, "merchants.csv"))
        self.load_locations(os.path.join(data_dir, "locations.csv"))
        self.load_devices_and_ips(
            os.path.join(data_dir, "devices.csv"),
            os.path.join(data_dir, "ip_addresses.csv")
        )
        
        # Load transactions (this takes the longest)
        self.load_transactions(os.path.join(data_dir, "all_transactions.csv"))
        
        # Create relationships
        self.create_transaction_relationships()
        self.create_user_merchant_relationships()
        self.create_user_location_relationships()
        self.create_fraud_patterns()
        
        print("Database setup complete!")

def main():
    parser = argparse.ArgumentParser(description='Set up Neo4j database for credit card fraud detection')
    parser.add_argument('--uri', type=str, default='bolt://localhost:7687',
                        help='Neo4j URI')
    parser.add_argument('--username', type=str, default='neo4j',
                        help='Neo4j username')
    parser.add_argument('--password', type=str, default='password',
                        help='Neo4j password')
    parser.add_argument('--data-dir', type=str, default='/home/ubuntu/credit_card_fraud_detection/data',
                        help='Directory containing data CSV files')
    
    args = parser.parse_args()
    
    neo4j = Neo4jFraudDetection(args.uri, args.username, args.password)
    
    try:
        neo4j.setup_database(args.data_dir)
    except Exception as e:
        print(f"Error setting up database: {e}")
    finally:
        neo4j.close()

if __name__ == "__main__":
    main()
