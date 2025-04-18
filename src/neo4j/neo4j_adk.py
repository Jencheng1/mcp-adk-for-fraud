#!/usr/bin/env python3
"""
Neo4j Query API for Credit Card Fraud Detection

This script provides an API for querying the Neo4j database for fraud detection.
It implements the Agent Development Kit (ADK) for Neo4j integration.
"""

import os
import json
import argparse
from neo4j import GraphDatabase

class Neo4jADK:
    """Agent Development Kit for Neo4j integration in fraud detection"""
    
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
    
    # Transaction Analysis Queries
    
    def get_transaction_by_id(self, transaction_id):
        """Get transaction details by ID"""
        query = """
        MATCH (t:Transaction {transactionId: $transaction_id})
        OPTIONAL MATCH (c:Card)-[:MADE]->(t)
        OPTIONAL MATCH (t)-[:AT]->(m:Merchant)
        OPTIONAL MATCH (t)-[:IN]->(l:Location)
        OPTIONAL MATCH (t)-[:USING]->(d:Device)
        OPTIONAL MATCH (d)-[:FROM]->(ip:IPAddress)
        OPTIONAL MATCH (u:User)-[:OWNS]->(c)
        RETURN t, c, m, l, d, ip, u
        """
        result = self.run_query(query, {"transaction_id": transaction_id})
        if not result:
            return None
        
        record = result[0]
        return {
            "transaction": record["t"],
            "card": record["c"],
            "merchant": record["m"],
            "location": record["l"],
            "device": record["d"],
            "ip_address": record["ip"],
            "user": record["u"]
        }
    
    def get_user_transactions(self, user_id, limit=100):
        """Get recent transactions for a user"""
        query = """
        MATCH (u:User {userId: $user_id})-[:OWNS]->(c:Card)-[:MADE]->(t:Transaction)
        OPTIONAL MATCH (t)-[:AT]->(m:Merchant)
        OPTIONAL MATCH (t)-[:IN]->(l:Location)
        RETURN t, c, m, l
        ORDER BY t.timestamp DESC
        LIMIT $limit
        """
        result = self.run_query(query, {"user_id": user_id, "limit": limit})
        return [
            {
                "transaction": record["t"],
                "card": record["c"],
                "merchant": record["m"],
                "location": record["l"]
            }
            for record in result
        ]
    
    def get_card_transactions(self, card_id, limit=100):
        """Get recent transactions for a card"""
        query = """
        MATCH (c:Card {cardId: $card_id})-[:MADE]->(t:Transaction)
        OPTIONAL MATCH (t)-[:AT]->(m:Merchant)
        OPTIONAL MATCH (t)-[:IN]->(l:Location)
        RETURN t, c, m, l
        ORDER BY t.timestamp DESC
        LIMIT $limit
        """
        result = self.run_query(query, {"card_id": card_id, "limit": limit})
        return [
            {
                "transaction": record["t"],
                "card": record["c"],
                "merchant": record["m"],
                "location": record["l"]
            }
            for record in result
        ]
    
    # Pattern Detection Queries
    
    def detect_velocity_anomalies(self, time_window_seconds=300, min_transactions=3):
        """Detect velocity anomalies (multiple transactions in short time)"""
        query = """
        MATCH (c:Card)-[:MADE]->(t1:Transaction)-[r:FOLLOWED_BY]->(t2:Transaction)
        WHERE r.timeDifference <= $time_window
        WITH c, t1, collect(t2) as subsequent_txs
        WHERE size(subsequent_txs) >= $min_transactions - 1
        MATCH (u:User)-[:OWNS]->(c)
        RETURN u.userId as user_id, c.cardId as card_id, 
               t1.transactionId as first_transaction_id,
               [tx in subsequent_txs | tx.transactionId] as subsequent_transaction_ids,
               size(subsequent_txs) + 1 as transaction_count
        ORDER BY transaction_count DESC
        """
        result = self.run_query(query, {
            "time_window": time_window_seconds,
            "min_transactions": min_transactions
        })
        return [dict(record) for record in result]
    
    def detect_location_anomalies(self, max_time_diff_seconds=7200, min_distance_km=500):
        """Detect location anomalies (transactions in distant locations in short time)"""
        query = """
        MATCH (c:Card)-[:MADE]->(t1:Transaction)-[:IN]->(l1:Location)
        MATCH (c)-[:MADE]->(t2:Transaction)-[:IN]->(l2:Location)
        WHERE t1.transactionId <> t2.transactionId
          AND duration.between(t1.timestamp, t2.timestamp).seconds <= $max_time_diff
          AND point.distance(l1.point, l2.point) / 1000 >= $min_distance
        MATCH (u:User)-[:OWNS]->(c)
        RETURN u.userId as user_id, c.cardId as card_id,
               t1.transactionId as transaction1_id, t2.transactionId as transaction2_id,
               t1.timestamp as timestamp1, t2.timestamp as timestamp2,
               l1.city as city1, l2.city as city2,
               l1.country as country1, l2.country as country2,
               round(point.distance(l1.point, l2.point) / 1000) as distance_km,
               duration.between(t1.timestamp, t2.timestamp).seconds as time_diff_seconds
        ORDER BY distance_km DESC
        """
        result = self.run_query(query, {
            "max_time_diff": max_time_diff_seconds,
            "min_distance": min_distance_km
        })
        return [dict(record) for record in result]
    
    def detect_unusual_merchant_activity(self):
        """Detect transactions at merchants not typically visited by the user"""
        query = """
        MATCH (u:User)-[:OWNS]->(c:Card)-[:MADE]->(t:Transaction)-[:AT]->(m:Merchant)
        WHERE NOT (u)-[:FREQUENTLY_VISITS]->(m)
          AND t.amount > 3 * toFloat(u.typicalTransactionAmount)
        RETURN u.userId as user_id, c.cardId as card_id,
               t.transactionId as transaction_id, t.amount as amount,
               m.merchantId as merchant_id, m.name as merchant_name,
               m.category as merchant_category, m.riskScore as merchant_risk_score
        ORDER BY t.amount DESC
        """
        result = self.run_query(query)
        return [dict(record) for record in result]
    
    def detect_high_risk_devices(self):
        """Detect transactions from high-risk devices"""
        query = """
        MATCH (c:Card)-[:MADE]->(t:Transaction)-[:USING]->(d:Device)-[:FROM]->(ip:IPAddress)
        WHERE d.riskScore > 0.7 OR ip.riskScore > 0.7
          OR ip.isProxy = true OR ip.isVpn = true OR ip.isTor = true
        MATCH (u:User)-[:OWNS]->(c)
        RETURN u.userId as user_id, c.cardId as card_id,
               t.transactionId as transaction_id, t.amount as amount,
               d.deviceId as device_id, d.deviceType as device_type,
               d.riskScore as device_risk_score,
               ip.ip as ip_address, ip.country as ip_country,
               ip.isProxy as is_proxy, ip.isVpn as is_vpn, ip.isTor as is_tor,
               ip.riskScore as ip_risk_score
        ORDER BY ip.riskScore + d.riskScore DESC
        """
        result = self.run_query(query)
        return [dict(record) for record in result]
    
    # Investigation Queries
    
    def get_transaction_chain(self, transaction_id, depth=3):
        """Get the chain of transactions before and after a given transaction"""
        query = """
        MATCH (t:Transaction {transactionId: $transaction_id})
        MATCH path = (prev)-[:FOLLOWED_BY*1..$depth]->(t)-[:FOLLOWED_BY*0..$depth]->(next)
        WHERE prev:Transaction AND next:Transaction
        WITH nodes(path) as transactions
        UNWIND transactions as tx
        MATCH (c:Card)-[:MADE]->(tx)
        OPTIONAL MATCH (tx)-[:AT]->(m:Merchant)
        OPTIONAL MATCH (tx)-[:IN]->(l:Location)
        RETURN tx.transactionId as transaction_id, tx.timestamp as timestamp,
               tx.amount as amount, tx.isFraudulent as is_fraudulent,
               c.cardId as card_id, m.name as merchant_name,
               l.city as city, l.country as country
        ORDER BY tx.timestamp
        """
        result = self.run_query(query, {"transaction_id": transaction_id, "depth": depth})
        return [dict(record) for record in result]
    
    def get_similar_fraud_patterns(self, transaction_id):
        """Get similar fraud patterns for a transaction"""
        query = """
        MATCH (t:Transaction {transactionId: $transaction_id})
        OPTIONAL MATCH (t)-[r:SIMILAR_TO]->(fp:FraudPattern)
        WITH t, fp, r
        
        // If no direct pattern match, find similar transactions
        OPTIONAL MATCH (t2:Transaction)-[:SIMILAR_TO]->(fp2:FraudPattern)
        WHERE t2.isFraudulent = true
          AND (t2.amount > t.amount * 0.8 AND t2.amount < t.amount * 1.2)
          AND t2.transactionId <> t.transactionId
        
        RETURN CASE WHEN fp IS NOT NULL THEN fp.patternId ELSE fp2.patternId END as pattern_id,
               CASE WHEN fp IS NOT NULL THEN fp.description ELSE fp2.description END as description,
               CASE WHEN fp IS NOT NULL THEN fp.patternType ELSE fp2.patternType END as pattern_type,
               CASE WHEN r IS NOT NULL THEN r.similarityScore ELSE 0.5 END as similarity_score
        ORDER BY similarity_score DESC
        LIMIT 5
        """
        result = self.run_query(query, {"transaction_id": transaction_id})
        return [dict(record) for record in result]
    
    def get_user_fraud_risk(self, user_id):
        """Calculate fraud risk for a user based on various factors"""
        query = """
        MATCH (u:User {userId: $user_id})
        
        // Get user's fraudulent transactions
        OPTIONAL MATCH (u)-[:OWNS]->(c:Card)-[:MADE]->(t:Transaction)
        WHERE t.isFraudulent = true
        WITH u, count(t) as fraud_count
        
        // Get user's high-risk devices
        OPTIONAL MATCH (u)-[:OWNS]->(c:Card)-[:MADE]->(t:Transaction)-[:USING]->(d:Device)
        WHERE d.riskScore > 0.7
        WITH u, fraud_count, count(DISTINCT d) as high_risk_devices
        
        // Get user's suspicious locations
        OPTIONAL MATCH (u)-[:OWNS]->(c:Card)-[:MADE]->(t:Transaction)-[:IN]->(l:Location)
        WHERE l.riskScore > 0.7
        WITH u, fraud_count, high_risk_devices, count(DISTINCT l) as suspicious_locations
        
        // Calculate overall risk score
        RETURN u.userId as user_id, u.name as name, u.email as email,
               fraud_count, high_risk_devices, suspicious_locations,
               u.riskScore as base_risk_score,
               CASE 
                 WHEN fraud_count > 0 THEN 0.9
                 WHEN high_risk_devices > 2 THEN 0.7
                 WHEN suspicious_locations > 3 THEN 0.6
                 ELSE u.riskScore
               END as calculated_risk_score
        """
        result = self.run_query(query, {"user_id": user_id})
        return dict(result[0]) if result else None
    
    # GraphRAG Knowledge Retrieval
    
    def retrieve_fraud_knowledge(self, transaction_properties, limit=5):
        """Retrieve relevant knowledge for fraud investigation using GraphRAG approach"""
        # Extract key properties from the transaction
        amount = transaction_properties.get("amount", 0)
        is_online = transaction_properties.get("is_online", False)
        merchant_category = transaction_properties.get("merchant_category", "")
        
        query = """
        // Find similar fraudulent transactions
        MATCH (t:Transaction)
        WHERE t.isFraudulent = true
          AND t.amount >= $amount * 0.8 AND t.amount <= $amount * 1.2
          AND t.isOnline = $is_online
        
        // Get connected entities
        MATCH (c:Card)-[:MADE]->(t)
        MATCH (t)-[:AT]->(m:Merchant)
        MATCH (t)-[:IN]->(l:Location)
        OPTIONAL MATCH (t)-[:USING]->(d:Device)
        OPTIONAL MATCH (t)-[:SIMILAR_TO]->(fp:FraudPattern)
        
        // Filter by merchant category if provided
        WITH t, c, m, l, d, fp
        WHERE $merchant_category = '' OR m.category = $merchant_category
        
        // Return relevant knowledge
        RETURN t.transactionId as transaction_id,
               t.amount as amount,
               t.timestamp as timestamp,
               m.category as merchant_category,
               l.country as country,
               CASE WHEN d IS NOT NULL THEN d.deviceType ELSE 'unknown' END as device_type,
               CASE WHEN fp IS NOT NULL THEN fp.description ELSE 'unknown' END as fraud_pattern,
               CASE WHEN fp IS NOT NULL THEN fp.patternType ELSE 'unknown' END as pattern_type
        ORDER BY t.timestamp DESC
        LIMIT $limit
        """
        
        result = self.run_query(query, {
            "amount": amount,
            "is_online": is_online,
            "merchant_category": merchant_category,
            "limit": limit
        })
        
        return [dict(record) for record in result]
    
    # Multi-Agent Support Functions
    
    def update_transaction_fraud_status(self, transaction_id, is_fraudulent, fraud_score, fraud_pattern=None):
        """Update a transaction's fraud status (used by Decision Agent)"""
        query = """
        MATCH (t:Transaction {transactionId: $transaction_id})
        SET t.isFraudulent = $is_fraudulent,
            t.fraudScore = $fraud_score,
            t.fraudPattern = $fraud_pattern,
            t.lastUpdated = datetime()
        
        WITH t
        WHERE $is_fraudulent = true AND $fraud_pattern IS NOT NULL
        
        // Connect to fraud pattern if fraudulent
        OPTIONAL MATCH (fp:FraudPattern)
        WHERE fp.patternType = $fraud_pattern
        WITH t, fp
        WHERE fp IS NOT NULL
        
        MERGE (t)-[r:SIMILAR_TO]->(fp)
        SET r.similarityScore = $fraud_score,
            r.matchedFeatures = ["amount", "location", "timing"]
        
        RETURN t.transactionId as transaction_id,
               t.isFraudulent as is_fraudulent,
               t.fraudScore as fraud_score,
               t.fraudPattern as fraud_pattern
        """
        
        result = self.run_query(query, {
            "transaction_id": transaction_id,
            "is_fraudulent": is_fraudulent,
            "fraud_score": fraud_score,
            "fraud_pattern": fraud_pattern
        })
        
        return dict(result[0]) if result else None
    
    def create_fraud_alert(self, transaction_id, alert_type, severity, description):
        """Create a fraud alert for a transaction (used by Alert Generation Agent)"""
        alert_id = f"alert_{transaction_id}"
        
        query = """
        MATCH (t:Transaction {transactionId: $transaction_id})
        
        // Create alert
        MERGE (a:Alert {alertId: $alert_id})
        SET a.timestamp = datetime(),
            a.severity = $severity,
            a.type = $alert_type,
            a.description = $description,
            a.isResolved = false
        
        // Connect alert to transaction
        MERGE (a)-[:FLAGGED]->(t)
        MERGE (t)-[:TRIGGERED]->(a)
        
        RETURN a.alertId as alert_id,
               a.timestamp as timestamp,
               a.severity as severity,
               a.type as alert_type,
               a.description as description,
               t.transactionId as transaction_id
        """
        
        result = self.run_query(query, {
            "transaction_id": transaction_id,
            "alert_id": alert_id,
            "severity": severity,
            "alert_type": alert_type,
            "description": description
        })
        
        return dict(result[0]) if result else None
    
    def get_user_profile(self, user_id):
        """Get comprehensive user profile for fraud analysis (used by User Profile Agent)"""
        query = """
        MATCH (u:User {userId: $user_id})
        
        // Get cards
        OPTIONAL MATCH (u)-[:OWNS]->(c:Card)
        
        // Get frequent merchants
        OPTIONAL MATCH (u)-[fv:FREQUENTLY_VISITS]->(m:Merchant)
        
        // Get typical locations
        OPTIONAL MATCH (u)-[ti:TYPICALLY_IN]->(l:Location)
        
        // Get recent transactions
        OPTIONAL MATCH (u)-[:OWNS]->(card:Card)-[:MADE]->(t:Transaction)
        WITH u, collect(DISTINCT c) as cards,
             collect(DISTINCT {merchant: m, frequency: fv.frequency}) as frequent_merchants,
             collect(DISTINCT {location: l, frequency: ti.frequency}) as typical_locations,
             collect(DISTINCT t) as transactions
        
        // Get transaction statistics
        WITH u, cards, frequent_merchants, typical_locations,
             [tx in transactions WHERE tx.timestamp > datetime() - duration({days: 30})] as recent_transactions
        
        RETURN u.userId as user_id,
               u.name as name,
               u.email as email,
               u.riskScore as risk_score,
               u.fraudHistory as fraud_history,
               u.segment as segment,
               u.typicalTransactionAmount as typical_amount,
               cards,
               frequent_merchants,
               typical_locations,
               size(recent_transactions) as transaction_count_30d,
               CASE WHEN size(recent_transactions) > 0 
                    THEN reduce(total = 0.0, tx IN recent_transactions | total + tx.amount) / size(recent_transactions)
                    ELSE 0.0
               END as avg_transaction_amount_30d
        """
        
        result = self.run_query(query, {"user_id": user_id})
        return dict(result[0]) if result else None

def main():
    parser = argparse.ArgumentParser(description='Neo4j ADK for credit card fraud detection')
    parser.add_argument('--uri', type=str, default='bolt://localhost:7687',
                        help='Neo4j URI')
    parser.add_argument('--username', type=str, default='neo4j',
                        help='Neo4j username')
    parser.add_argument('--password', type=str, default='password',
                        help='Neo4j password')
    parser.add_argument('--query', type=str, choices=[
        'velocity', 'location', 'merchant', 'device', 'user_risk'
    ], default='velocity', help='Query to run')
    parser.add_argument('--param', type=str, default=None,
                        help='Parameter for the query (e.g., user_id, transaction_id)')
    
    args = parser.parse_args()
    
    adk = Neo4jADK(args.uri, args.username, args.password)
    
    try:
        # Run the selected query
        if args.query == 'velocity':
            result = adk.detect_velocity_anomalies()
        elif args.query == 'location':
            result = adk.detect_location_anomalies()
        elif args.query == 'merchant':
            result = adk.detect_unusual_merchant_activity()
        elif args.query == 'device':
            result = adk.detect_high_risk_devices()
        elif args.query == 'user_risk' and args.param:
            result = adk.get_user_fraud_risk(args.param)
        else:
            result = {"error": "Invalid query or missing parameter"}
        
        # Print the result
        print(json.dumps(result, indent=2, default=str))
    
    except Exception as e:
        print(f"Error running query: {e}")
    finally:
        adk.close()

if __name__ == "__main__":
    main()
