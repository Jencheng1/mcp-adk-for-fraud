# Credit Card Fraud Detection System Documentation

## Overview

This document provides comprehensive documentation for the end-to-end credit card fraud detection system. The system integrates multiple advanced technologies to detect and investigate fraudulent credit card transactions in real-time, with a special focus on multi-agent and multi-modal approaches using Model Context Protocol (MCP), Agent-to-Agent (A2A) communication, and Agent Development Kit (ADK) for Neo4j integration.

## System Architecture

The system architecture is designed to process credit card transactions in real-time, detect potential fraud patterns, and provide tools for investigation and analysis. The architecture consists of the following key components:

1. **Data Ingestion Layer**: Kafka for real-time transaction streaming
2. **Processing Layer**: PySpark for large-scale data processing
3. **Storage Layer**: Neo4j graph database for storing transaction data and relationships
4. **Detection Layer**: Graph Neural Networks (GNN) for pattern recognition
5. **Multi-Agent System**: LangGraph for orchestrating multiple specialized agents
6. **Knowledge Retrieval**: GraphRAG for retrieving relevant information during investigations
7. **Visualization Layer**: Interactive dashboards for monitoring and investigation

## Technologies Used

### Core Technologies

- **Kafka**: Real-time data streaming platform for transaction ingestion
- **PySpark**: Distributed computing framework for large-scale data processing
- **Neo4j**: Graph database for storing transaction data and relationships
- **Graph Neural Networks (GNN)**: Deep learning models for graph-structured data
- **LangGraph**: Framework for building multi-agent systems
- **GraphRAG**: Graph-based retrieval augmented generation for knowledge retrieval

### Multi-Agent Integration Technologies

- **Model Context Protocol (MCP)**: Protocol for maintaining context across different agents
- **Agent-to-Agent (A2A) Communication**: Framework for structured message passing between agents
- **Agent Development Kit (ADK)**: Toolkit for integrating agents with Neo4j

## Multi-Agent System Architecture

The multi-agent system is the core of the fraud detection pipeline, consisting of specialized agents that work together to detect and investigate fraudulent transactions.

### Agent Types

1. **Transaction Analysis Agent**: Processes incoming transactions and performs initial analysis
2. **User Profile Agent**: Analyzes user behavior patterns
3. **Merchant Risk Agent**: Assesses merchant risk levels
4. **Pattern Detection Agent**: Identifies patterns across multiple transactions
5. **Investigation Agent**: Conducts deeper investigation of suspicious activities
6. **Decision Agent**: Makes final fraud determination
7. **Feedback Collection Agent**: Gathers feedback on detection accuracy
8. **Learning Agent**: Updates models based on feedback

### Model Context Protocol (MCP)

The Model Context Protocol enables the system to maintain context across different agents:

1. Each agent creates a **context** for its analysis
2. Contexts can contain **structured data**, **text**, and **references** to other contexts
3. Contexts are **linked** to form a knowledge graph
4. The MCP allows for **multi-modal** context handling (text, structured data, images)
5. Agents can **merge contexts** to create a comprehensive view

Example MCP Context:
```json
{
    "context_id": "ctx_tx_123456",
    "context_type": "transaction_analysis",
    "created_at": "2025-04-18T12:34:56.789Z",
    "content": {
        "transaction": {
            "transaction_id": "tx_123456",
            "amount": 1299.99,
            "timestamp": "2025-04-18T12:34:56.789Z",
            "merchant_id": "merch_789",
            "user_id": "user_456"
        },
        "analysis_results": {
            "initial_fraud_score": 0.75,
            "fraud_indicators": ["high_amount", "unusual_merchant"]
        }
    },
    "metadata": {
        "agent": "transaction_analysis_agent",
        "processing_time": 0.123
    },
    "references": ["ctx_user_456"]
}
```

### Agent-to-Agent (A2A) Communication

The Agent-to-Agent communication system enables structured message passing between agents:

1. Agents send **messages** with specific message types
2. Messages contain **content** relevant to the task
3. Messages can reference **contexts** from the MCP
4. Agents can **subscribe** to receive messages from other agents
5. The system supports both **direct messaging** and **broadcasting**

Example A2A Message:
```json
{
    "message_id": "msg_000123",
    "sender": "pattern_detection_agent",
    "receiver": "investigation_agent",
    "message_type": "patterns_detected",
    "priority": "high",
    "content": {
        "transaction_id": "tx_123456",
        "patterns_detected": ["velocity_anomaly", "location_anomaly"],
        "pattern_score": 0.65,
        "combined_score": 0.85
    },
    "timestamp": "2025-04-18T12:35:01.234Z",
    "context_references": ["ctx_tx_123456", "ctx_pattern_123456"],
    "status": "sent"
}
```

### Agent Development Kit (ADK) for Neo4j

The Agent Development Kit (ADK) provides specialized capabilities for fraud detection using Neo4j:

#### Transaction Analysis
- `get_transaction_by_id`: Get detailed transaction information
- `get_user_transactions`: Get recent transactions for a user
- `get_card_transactions`: Get recent transactions for a card

#### Pattern Detection
- `detect_velocity_anomalies`: Detect multiple transactions in short time
- `detect_location_anomalies`: Detect transactions in distant locations
- `detect_unusual_merchant_activity`: Detect transactions at unusual merchants
- `detect_high_risk_devices`: Detect transactions from suspicious devices

#### Investigation
- `get_transaction_chain`: Get chain of related transactions
- `get_similar_fraud_patterns`: Find similar fraud patterns
- `get_user_fraud_risk`: Calculate user's fraud risk

#### GraphRAG Knowledge Retrieval
- `retrieve_fraud_knowledge`: Retrieve relevant knowledge for investigation

#### Multi-Agent Support
- `update_transaction_fraud_status`: Update transaction status
- `create_fraud_alert`: Create alerts for fraudulent transactions
- `get_user_profile`: Get comprehensive user profile

## Data Flow

The data flow in the system follows these steps:

1. **Transaction Ingestion**: Credit card transactions are ingested in real-time through Kafka
2. **Initial Processing**: PySpark processes the incoming transactions and performs initial feature extraction
3. **Graph Storage**: Transactions and related entities are stored in Neo4j graph database
4. **Multi-Agent Analysis**:
   - Transaction Analysis Agent processes the transaction and creates initial context
   - User Profile Agent and Merchant Risk Agent provide additional context
   - Pattern Detection Agent identifies potential fraud patterns
   - Investigation Agent conducts deeper investigation if needed
   - Decision Agent makes final fraud determination
5. **Feedback Loop**: Feedback is collected and used to improve the models
6. **Visualization**: Results are displayed in real-time dashboards

## Neo4j Graph Schema

The Neo4j graph database schema captures the relationships between entities:

1. **Users** own **Cards** which make **Transactions**
2. Transactions occur at **Merchants** and in **Locations**
3. Online transactions use **Devices** with **IP Addresses**
4. Transactions can be linked to **Fraud Patterns**
5. The graph captures temporal relationships between transactions (**FOLLOWED_BY**)
6. User behavior patterns are captured (**FREQUENTLY_VISITS**, **TYPICALLY_IN**)

## GNN Model Architecture

The Graph Neural Network (GNN) model is used to detect fraud patterns in the transaction graph:

1. **Input Layer**: Takes transaction graph with node and edge features
2. **Graph Convolutional Layers**: Extract features from the graph structure
3. **Attention Mechanism**: Focuses on important nodes and edges
4. **Pooling Layer**: Aggregates node features
5. **Fully Connected Layers**: Produce final fraud score
6. **Output Layer**: Classifies transaction as fraudulent or legitimate

## Dashboards

The system includes two main dashboards:

### Transaction Monitoring Dashboard

- Real-time transaction feed
- Fraud metrics and statistics
- Fraud pattern visualization
- Multi-agent system visualization
- ADK visualization

### Fraud Investigation Dashboard

- Fraud pattern analysis
- Transaction investigation interface
- Multi-agent system visualization
- Agent logs
- MCP context visualization
- A2A message visualization
- ADK capabilities demonstration

## Fraud Patterns Detected

The system can detect the following fraud patterns:

1. **Card Testing**: Small transactions followed by larger ones
2. **Identity Theft**: Transactions inconsistent with user profile
3. **Account Takeover**: Sudden change in transaction patterns
4. **Card-not-present Fraud**: Online transactions with suspicious patterns
5. **Merchant Fraud**: Transactions at high-risk merchants
6. **Application Fraud**: New accounts with suspicious activity
7. **Transaction Laundering**: Complex chains of transactions
8. **Velocity Abuse**: Multiple transactions in short time
9. **Location Anomaly**: Transactions in distant locations in short time
10. **Amount Anomaly**: Transactions with unusually high amounts

## Installation and Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.8+
- Java 11+
- Neo4j 4.4+

### Installation Steps

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Start Kafka: `docker-compose -f src/kafka/docker-compose.yml up -d`
4. Set up Neo4j database: `python src/neo4j/setup_database.py`
5. Generate test data: `python data/generate_test_data.py`
6. Start the fraud detection pipeline: `python src/multi_agent/fraud_detection_agents.py`
7. Start the dashboards:
   - Transaction Monitoring: `streamlit run src/dashboard/transaction_monitoring.py`
   - Fraud Investigation: `streamlit run src/dashboard/fraud_investigation.py`

## Conclusion

This end-to-end credit card fraud detection system demonstrates the power of combining multiple advanced technologies, particularly the integration of multi-agent and multi-modal approaches using Model Context Protocol (MCP), Agent-to-Agent (A2A) communication, and Agent Development Kit (ADK) for Neo4j. The system provides real-time fraud detection with high accuracy and comprehensive investigation capabilities.

## Future Enhancements

1. Integration with payment gateways for real-time intervention
2. Enhanced multi-modal capabilities (image analysis, voice recognition)
3. Expanded agent types for more specialized analysis
4. Federated learning across multiple financial institutions
5. Advanced explainable AI features for regulatory compliance
