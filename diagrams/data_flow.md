# Data Flow for Real-Time Transaction Processing

## Overview

This document outlines the data flow for real-time transaction processing in the credit card fraud detection system, focusing on how transaction data moves through the system from ingestion to fraud determination.

## Data Flow Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Transaction    │     │  Kafka          │     │  PySpark        │     │  Feature        │
│  Sources        │────▶│  Topics         │────▶│  Streaming      │────▶│  Extraction     │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
                                                                                │
                                                                                ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Fraud          │     │  Multi-Agent    │     │  Neo4j Graph    │     │  Enrichment     │
│  Determination  │◀────│  Analysis       │◀────│  Database       │◀────│  Service        │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
        │                                                                        ▲
        ▼                                                                        │
┌─────────────────┐     ┌─────────────────┐                          ┌─────────────────┐
│  Alert          │     │  Transaction    │                          │  External       │
│  Generation     │     │  Disposition    │                          │  Data Sources   │
└─────────────────┘     └─────────────────┘                          └─────────────────┘
```

## Detailed Process Flow

### 1. Transaction Ingestion

**Sources:**
- Point of Sale (POS) terminals
- Online payment gateways
- Mobile payment applications
- ATM transactions
- Contactless payments

**Data Format:**
```json
{
  "transaction_id": "tx_123456789",
  "timestamp": "2025-04-18T17:30:00Z",
  "card_id": "card_987654321",
  "merchant_id": "merch_123456",
  "amount": 299.99,
  "currency": "USD",
  "transaction_type": "purchase",
  "payment_method": "chip",
  "location": {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "country": "US",
    "city": "New York"
  },
  "device_info": {
    "ip_address": "192.168.1.1",
    "device_id": "device_123456",
    "user_agent": "Mozilla/5.0..."
  }
}
```

### 2. Kafka Streaming

**Topics:**
- `raw-transactions`: Raw transaction events
- `enriched-transactions`: Transactions with additional context
- `fraud-alerts`: Detected fraud events
- `transaction-dispositions`: Final transaction decisions

**Partitioning Strategy:**
- Partition by card_id to ensure transaction order
- Multiple consumer groups for different processing needs

**Retention Policy:**
- Raw transactions: 7 days
- Enriched transactions: 30 days
- Fraud alerts: 90 days
- Transaction dispositions: 90 days

### 3. PySpark Streaming Processing

**Operations:**
- Deserialization of transaction data
- Initial validation and filtering
- Batch processing for feature extraction
- Stream-to-stream joins with reference data
- Windowed aggregations for pattern detection

**Processing Latency:**
- Target: < 100ms from ingestion to feature extraction
- Micro-batch processing with 1-second intervals

**Scalability:**
- Dynamic scaling based on transaction volume
- Resource allocation based on time of day patterns

### 4. Feature Extraction

**Transaction Features:**
- Amount normalization
- Time-of-day encoding
- Day-of-week encoding
- Geographic distance from previous transactions
- Velocity checks (transactions per hour/day)
- Merchant category features

**User Features:**
- Historical spending patterns
- Average transaction amount
- Transaction frequency
- Common merchants
- Geographic spread of transactions

**Merchant Features:**
- Fraud rate
- Average transaction amount
- Transaction volume
- Business category
- Operating hours

### 5. Data Enrichment

**External Data Sources:**
- User profile database
- Merchant reputation database
- IP geolocation service
- Device fingerprinting service
- Fraud blacklists

**Enrichment Process:**
- Lookup user history and profile
- Retrieve merchant risk score
- Validate location information
- Check device against known fraud devices
- Add contextual information about transaction

**Enriched Data Format:**
```json
{
  "transaction_id": "tx_123456789",
  "timestamp": "2025-04-18T17:30:00Z",
  "card_id": "card_987654321",
  "user_id": "user_123456",
  "merchant_id": "merch_123456",
  "amount": 299.99,
  "currency": "USD",
  "transaction_type": "purchase",
  "payment_method": "chip",
  "location": {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "country": "US",
    "city": "New York"
  },
  "device_info": {
    "ip_address": "192.168.1.1",
    "device_id": "device_123456",
    "user_agent": "Mozilla/5.0...",
    "risk_score": 0.15
  },
  "user_profile": {
    "risk_score": 0.05,
    "account_age_days": 730,
    "typical_transaction_amount": 150.25,
    "typical_merchants": ["merch_111", "merch_222", "merch_333"],
    "typical_locations": ["New York", "Boston"]
  },
  "merchant_profile": {
    "risk_score": 0.02,
    "category": "Electronics",
    "average_transaction": 275.50,
    "fraud_rate": 0.001
  },
  "context_features": {
    "amount_vs_average": 1.99,
    "distance_from_last_transaction_km": 2.5,
    "time_since_last_transaction_hours": 48.5,
    "is_common_merchant": false,
    "is_common_location": true
  }
}
```

### 6. Neo4j Graph Database Storage

**Node Types:**
- User
- Card
- Transaction
- Merchant
- Device
- Location
- IP Address

**Relationship Types:**
- (User)-[:OWNS]->(Card)
- (Card)-[:USED_IN]->(Transaction)
- (Transaction)-[:AT]->(Merchant)
- (Transaction)-[:FROM]->(Location)
- (Transaction)-[:USING]->(Device)
- (Device)-[:HAS_IP]->(IP Address)
- (User)-[:FREQUENTLY_VISITS]->(Merchant)
- (User)-[:TYPICALLY_IN]->(Location)

**Graph Storage Process:**
- Create transaction node
- Create or update related entity nodes
- Establish relationships between nodes
- Update graph metrics and aggregations

### 7. Multi-Agent Analysis

**Agent Processing Flow:**
1. Transaction Analysis Agent evaluates individual transaction
2. User Profile Agent provides behavioral context
3. Merchant Risk Agent assesses merchant risk
4. Pattern Detection Agent identifies suspicious patterns
5. Investigation Agent performs deeper analysis if needed
6. Decision Agent makes final determination

**Agent Communication:**
- A2A protocol for message passing
- MCP for multi-modal context processing
- ADK for Neo4j graph queries and updates

**Analysis Output:**
```json
{
  "transaction_id": "tx_123456789",
  "fraud_score": 0.87,
  "confidence": 0.92,
  "analysis_results": [
    {
      "agent": "transaction_analysis",
      "score": 0.75,
      "reasons": ["amount_anomaly", "unusual_merchant"]
    },
    {
      "agent": "pattern_detection",
      "score": 0.95,
      "reasons": ["velocity_check_failed", "location_anomaly"]
    },
    {
      "agent": "investigation",
      "score": 0.90,
      "reasons": ["similar_to_known_fraud_pattern"]
    }
  ],
  "decision": "flag_for_review",
  "explanation": "Transaction flagged due to unusual amount for this user and suspicious pattern of transactions across multiple locations in short timeframe."
}
```

### 8. Fraud Determination

**Decision Types:**
- Approve: Transaction is legitimate
- Deny: Transaction is fraudulent
- Flag for Review: Requires human investigation
- Request Additional Authentication: Step-up authentication needed

**Decision Factors:**
- Fraud score from multi-agent analysis
- Transaction amount and risk
- User history and standing
- Merchant reputation
- Regulatory requirements

**Decision Process:**
1. Evaluate fraud score against thresholds
2. Apply business rules based on amount and risk
3. Consider user and merchant context
4. Determine appropriate action
5. Record decision and reasoning

### 9. Alert Generation

**Alert Types:**
- Real-time fraud alerts
- Suspicious activity notifications
- Pattern detection alerts
- Investigation requests

**Alert Channels:**
- Dashboard notifications
- Email alerts
- SMS notifications
- API webhooks to external systems

**Alert Format:**
```json
{
  "alert_id": "alert_123456",
  "timestamp": "2025-04-18T17:30:15Z",
  "severity": "high",
  "type": "fraud_detected",
  "transaction_id": "tx_123456789",
  "card_id": "card_987654321",
  "user_id": "user_123456",
  "fraud_score": 0.87,
  "reasons": [
    "amount_anomaly",
    "velocity_check_failed",
    "location_anomaly",
    "similar_to_known_fraud_pattern"
  ],
  "recommended_action": "block_card",
  "requires_investigation": true
}
```

### 10. Transaction Disposition

**Disposition Actions:**
- Record final decision
- Update transaction status
- Notify relevant systems
- Update user and merchant profiles
- Feed back to learning systems

**Disposition Flow:**
1. Record disposition in transaction database
2. Update Neo4j graph with disposition
3. Send disposition to Kafka topic
4. Update fraud models with feedback
5. Generate reports and analytics

## Performance Metrics

- **End-to-End Latency**: Time from transaction initiation to fraud determination
  - Target: < 500ms for 95% of transactions
  
- **Throughput**: Transactions processed per second
  - Target: 10,000+ TPS during peak periods
  
- **Accuracy Metrics**:
  - False Positive Rate: < 1%
  - False Negative Rate: < 0.1%
  - Precision: > 95%
  - Recall: > 99%

## Data Retention and Compliance

- Raw transaction data: 7 years (for regulatory compliance)
- Fraud alerts: 7 years
- Transaction dispositions: 7 years
- User profiles: Duration of relationship + 7 years
- Merchant profiles: Duration of relationship + 7 years

## Error Handling and Recovery

- **Data Validation Errors**: Log, quarantine, and alert
- **Processing Failures**: Retry with exponential backoff
- **System Outages**: Failover to backup systems
- **Data Inconsistencies**: Reconciliation processes
- **Recovery Procedures**: Automated and manual recovery workflows

## Monitoring and Observability

- **Metrics Collection**: Latency, throughput, error rates
- **Log Aggregation**: Centralized logging with structured data
- **Alerting**: Threshold-based alerts for anomalies
- **Dashboards**: Real-time monitoring of system health
- **Tracing**: Distributed tracing for end-to-end visibility
