#!/usr/bin/env python3
"""
Real-time Transaction Monitoring Dashboard for Credit Card Fraud Detection

This script implements a Streamlit dashboard for monitoring real-time credit card
transactions and fraud detection results.
"""

import os
import json
import time
import random
import argparse
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from neo4j import GraphDatabase

# Add parent directory to path for imports
import sys
sys.path.append('/home/ubuntu/credit_card_fraud_detection/src/neo4j')
from neo4j_adk import Neo4jADK

# Set page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection - Real-time Monitoring",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dashboard styles
st.markdown("""
<style>
    .fraud-alert {
        background-color: #ffcccb;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ff0000;
        margin-bottom: 10px;
    }
    .suspicious-alert {
        background-color: #ffffcc;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ffcc00;
        margin-bottom: 10px;
    }
    .normal-transaction {
        background-color: #e6ffe6;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #00cc00;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 15px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px;
        color: #555;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f3ff;
        border-bottom: 2px solid #4da6ff;
    }
</style>
""", unsafe_allow_html=True)

class TransactionDashboard:
    """Dashboard for monitoring credit card transactions and fraud detection"""
    
    def __init__(self, neo4j_uri, neo4j_username, neo4j_password, data_dir):
        """Initialize the dashboard"""
        self.neo4j_adk = Neo4jADK(neo4j_uri, neo4j_username, neo4j_password)
        self.data_dir = data_dir
        
        # Load transaction data
        self.load_data()
        
        # Initialize session state
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        if 'transaction_count' not in st.session_state:
            st.session_state.transaction_count = 0
        if 'fraud_count' not in st.session_state:
            st.session_state.fraud_count = 0
        if 'alert_count' not in st.session_state:
            st.session_state.alert_count = 0
        if 'recent_transactions' not in st.session_state:
            st.session_state.recent_transactions = []
        if 'fraud_by_pattern' not in st.session_state:
            st.session_state.fraud_by_pattern = {}
        if 'fraud_by_merchant' not in st.session_state:
            st.session_state.fraud_by_merchant = {}
        if 'fraud_by_location' not in st.session_state:
            st.session_state.fraud_by_location = {}
        if 'selected_transaction' not in st.session_state:
            st.session_state.selected_transaction = None
    
    def load_data(self):
        """Load transaction data from CSV files"""
        try:
            self.transactions_df = pd.read_csv(os.path.join(self.data_dir, 'all_transactions.csv'))
            self.users_df = pd.read_csv(os.path.join(self.data_dir, 'users.csv'))
            self.cards_df = pd.read_csv(os.path.join(self.data_dir, 'cards.csv'))
            self.merchants_df = pd.read_csv(os.path.join(self.data_dir, 'merchants.csv'))
            self.locations_df = pd.read_csv(os.path.join(self.data_dir, 'locations.csv'))
            
            # Convert timestamp to datetime
            self.transactions_df['timestamp'] = pd.to_datetime(self.transactions_df['timestamp'])
            
            # Sort by timestamp
            self.transactions_df = self.transactions_df.sort_values('timestamp')
            
            # Create a mapping of IDs to names for easier display
            self.merchant_names = dict(zip(self.merchants_df['merchant_id'], self.merchants_df['name']))
            self.location_names = dict(zip(self.locations_df['location_id'], 
                                          self.locations_df['city'] + ', ' + self.locations_df['country']))
            self.user_names = dict(zip(self.users_df['user_id'], self.users_df['name']))
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            self.transactions_df = pd.DataFrame()
            self.users_df = pd.DataFrame()
            self.cards_df = pd.DataFrame()
            self.merchants_df = pd.DataFrame()
            self.locations_df = pd.DataFrame()
    
    def simulate_real_time_data(self):
        """Simulate real-time transaction data"""
        # Get current time
        current_time = datetime.now()
        
        # Only update if at least 2 seconds have passed since last update
        if (current_time - st.session_state.last_update).total_seconds() < 2:
            return
        
        # Update last update time
        st.session_state.last_update = current_time
        
        # Simulate new transactions (1-3 new transactions)
        num_new_transactions = random.randint(1, 3)
        
        for _ in range(num_new_transactions):
            # Get a random transaction from the dataset
            tx_idx = random.randint(0, len(self.transactions_df) - 1)
            tx = self.transactions_df.iloc[tx_idx].copy()
            
            # Update timestamp to current time with some random offset
            tx['timestamp'] = current_time - timedelta(seconds=random.randint(0, 60))
            
            # Create transaction record
            transaction = {
                'transaction_id': tx['transaction_id'],
                'timestamp': tx['timestamp'],
                'amount': tx['amount'],
                'merchant_name': self.merchant_names.get(tx['merchant_id'], 'Unknown'),
                'merchant_category': tx['mcc'],
                'location': self.location_names.get(tx['location_id'], 'Unknown'),
                'user_name': self.user_names.get(tx['user_id'], 'Unknown'),
                'card_id': tx['card_id'],
                'is_online': tx['is_online'],
                'is_fraudulent': tx['is_fraudulent'],
                'fraud_score': tx['fraud_score'],
                'fraud_pattern': tx['fraud_pattern'] if tx['is_fraudulent'] else None
            }
            
            # Add to recent transactions
            st.session_state.recent_transactions.insert(0, transaction)
            
            # Keep only the most recent 100 transactions
            if len(st.session_state.recent_transactions) > 100:
                st.session_state.recent_transactions = st.session_state.recent_transactions[:100]
            
            # Update counters
            st.session_state.transaction_count += 1
            if tx['is_fraudulent']:
                st.session_state.fraud_count += 1
                st.session_state.alert_count += 1
                
                # Update fraud by pattern
                pattern = tx['fraud_pattern'] if pd.notna(tx['fraud_pattern']) else 'unknown'
                if pattern in st.session_state.fraud_by_pattern:
                    st.session_state.fraud_by_pattern[pattern] += 1
                else:
                    st.session_state.fraud_by_pattern[pattern] = 1
                
                # Update fraud by merchant
                merchant = self.merchant_names.get(tx['merchant_id'], 'Unknown')
                if merchant in st.session_state.fraud_by_merchant:
                    st.session_state.fraud_by_merchant[merchant] += 1
                else:
                    st.session_state.fraud_by_merchant[merchant] = 1
                
                # Update fraud by location
                location = self.location_names.get(tx['location_id'], 'Unknown')
                if location in st.session_state.fraud_by_location:
                    st.session_state.fraud_by_location[location] += 1
                else:
                    st.session_state.fraud_by_location[location] = 1
    
    def render_header(self):
        """Render the dashboard header"""
        st.title("Credit Card Fraud Detection - Real-time Monitoring")
        st.markdown("Monitor real-time credit card transactions and fraud detection results")
        
        # Add refresh button
        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("Refresh Data"):
                st.experimental_rerun()
    
    def render_metrics(self):
        """Render key metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:,}</div>
                <div class="metric-label">Transactions Processed</div>
            </div>
            """.format(st.session_state.transaction_count), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:,}</div>
                <div class="metric-label">Fraud Detected</div>
            </div>
            """.format(st.session_state.fraud_count), unsafe_allow_html=True)
        
        with col3:
            fraud_rate = (st.session_state.fraud_count / st.session_state.transaction_count * 100) if st.session_state.transaction_count > 0 else 0
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.2f}%</div>
                <div class="metric-label">Fraud Rate</div>
            </div>
            """.format(fraud_rate), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:,}</div>
                <div class="metric-label">Active Alerts</div>
            </div>
            """.format(st.session_state.alert_count), unsafe_allow_html=True)
    
    def render_transaction_feed(self):
        """Render real-time transaction feed"""
        st.subheader("Real-time Transaction Feed")
        
        # Create a container for the transaction feed
        feed_container = st.container()
        
        with feed_container:
            for i, tx in enumerate(st.session_state.recent_transactions[:10]):
                if tx['is_fraudulent']:
                    st.markdown(f"""
                    <div class="fraud-alert">
                        <strong>FRAUD ALERT:</strong> ${tx['amount']:.2f} at {tx['merchant_name']} ({tx['timestamp'].strftime('%H:%M:%S')})
                        <br>User: {tx['user_name']} | Card: {tx['card_id']} | Pattern: {tx['fraud_pattern'] or 'Unknown'}
                        <br>Score: {tx['fraud_score']:.2f}
                    </div>
                    """, unsafe_allow_html=True)
                elif tx['fraud_score'] > 0.3:
                    st.markdown(f"""
                    <div class="suspicious-alert">
                        <strong>SUSPICIOUS:</strong> ${tx['amount']:.2f} at {tx['merchant_name']} ({tx['timestamp'].strftime('%H:%M:%S')})
                        <br>User: {tx['user_name']} | Card: {tx['card_id']}
                        <br>Score: {tx['fraud_score']:.2f}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="normal-transaction">
                        <strong>NORMAL:</strong> ${tx['amount']:.2f} at {tx['merchant_name']} ({tx['timestamp'].strftime('%H:%M:%S')})
                        <br>User: {tx['user_name']} | Card: {tx['card_id']}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add a button to view transaction details
                if st.button(f"View Details", key=f"view_{i}"):
                    st.session_state.selected_transaction = tx
    
    def render_fraud_patterns_chart(self):
        """Render chart of fraud patterns"""
        st.subheader("Fraud by Pattern")
        
        if not st.session_state.fraud_by_pattern:
            st.info("No fraud patterns detected yet")
            return
        
        # Create dataframe for chart
        pattern_df = pd.DataFrame({
            'Pattern': list(st.session_state.fraud_by_pattern.keys()),
            'Count': list(st.session_state.fraud_by_pattern.values())
        })
        
        # Sort by count
        pattern_df = pattern_df.sort_values('Count', ascending=False)
        
        # Create bar chart
        fig = px.bar(
            pattern_df,
            x='Pattern',
            y='Count',
            color='Pattern',
            title='Fraud by Pattern'
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Fraud Pattern',
            yaxis_title='Number of Fraudulent Transactions',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_fraud_merchants_chart(self):
        """Render chart of fraud by merchant"""
        st.subheader("Fraud by Merchant")
        
        if not st.session_state.fraud_by_merchant:
            st.info("No fraudulent merchants detected yet")
            return
        
        # Create dataframe for chart
        merchant_df = pd.DataFrame({
            'Merchant': list(st.session_state.fraud_by_merchant.keys()),
            'Count': list(st.session_state.fraud_by_merchant.values())
        })
        
        # Sort by count and take top 10
        merchant_df = merchant_df.sort_values('Count', ascending=False).head(10)
        
        # Create bar chart
        fig = px.bar(
            merchant_df,
            x='Merchant',
            y='Count',
            color='Merchant',
            title='Top 10 Merchants with Fraud'
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Merchant',
            yaxis_title='Number of Fraudulent Transactions',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_fraud_locations_chart(self):
        """Render chart of fraud by location"""
        st.subheader("Fraud by Location")
        
        if not st.session_state.fraud_by_location:
            st.info("No fraudulent locations detected yet")
            return
        
        # Create dataframe for chart
        location_df = pd.DataFrame({
            'Location': list(st.session_state.fraud_by_location.keys()),
            'Count': list(st.session_state.fraud_by_location.values())
        })
        
        # Sort by count and take top 10
        location_df = location_df.sort_values('Count', ascending=False).head(10)
        
        # Create bar chart
        fig = px.bar(
            location_df,
            x='Location',
            y='Count',
            color='Location',
            title='Top 10 Locations with Fraud'
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Location',
            yaxis_title='Number of Fraudulent Transactions',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_transaction_details(self):
        """Render transaction details"""
        if not st.session_state.selected_transaction:
            return
        
        tx = st.session_state.selected_transaction
        
        st.subheader("Transaction Details")
        
        # Create columns for details
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Basic Information")
            st.markdown(f"**Transaction ID:** {tx['transaction_id']}")
            st.markdown(f"**Timestamp:** {tx['timestamp']}")
            st.markdown(f"**Amount:** ${tx['amount']:.2f}")
            st.markdown(f"**Merchant:** {tx['merchant_name']}")
            st.markdown(f"**Category:** {tx['merchant_category']}")
            st.markdown(f"**Location:** {tx['location']}")
            st.markdown(f"**Online Transaction:** {'Yes' if tx['is_online'] else 'No'}")
        
        with col2:
            st.markdown("### Fraud Analysis")
            st.markdown(f"**Fraud Score:** {tx['fraud_score']:.2f}")
            st.markdown(f"**Fraudulent:** {'Yes' if tx['is_fraudulent'] else 'No'}")
            if tx['is_fraudulent'] and tx['fraud_pattern']:
                st.markdown(f"**Fraud Pattern:** {tx['fraud_pattern']}")
            
            st.markdown("### User Information")
            st.markdown(f"**User:** {tx['user_name']}")
            st.markdown(f"**Card ID:** {tx['card_id']}")
        
        # Add a button to close details
        if st.button("Close Details"):
            st.session_state.selected_transaction = None
            st.experimental_rerun()
    
    def render_multi_agent_visualization(self):
        """Render visualization of multi-agent system"""
        st.subheader("Multi-Agent System Visualization")
        
        # Create tabs for different visualizations
        tabs = st.tabs(["Agent Interaction", "MCP Context Flow", "A2A Communication"])
        
        with tabs[0]:
            st.markdown("### Agent Interaction Diagram")
            
            # Create a diagram of agent interactions
            st.markdown("""
            ```mermaid
            graph TD
                TA[Transaction Analysis Agent] --> PD[Pattern Detection Agent]
                TA --> UP[User Profile Agent]
                TA --> MR[Merchant Risk Agent]
                UP --> PD
                MR --> PD
                PD --> IA[Investigation Agent]
                PD --> DA[Decision Agent]
                IA --> DA
                DA --> FC[Feedback Collection Agent]
                FC --> LA[Learning Agent]
            ```
            """)
            
            st.markdown("""
            This diagram shows how the different agents in the system interact with each other:
            
            1. **Transaction Analysis Agent** processes incoming transactions and performs initial analysis
            2. **User Profile Agent** analyzes user behavior patterns
            3. **Merchant Risk Agent** assesses merchant risk levels
            4. **Pattern Detection Agent** identifies patterns across multiple transactions
            5. **Investigation Agent** conducts deeper investigation of suspicious activities
            6. **Decision Agent** makes final fraud determination
            7. **Feedback Collection Agent** gathers feedback on detection accuracy
            8. **Learning Agent** updates models based on feedback
            """)
        
        with tabs[1]:
            st.markdown("### Model Context Protocol (MCP) Flow")
            
            # Create a visualization of MCP context flow
            st.markdown("""
            ```mermaid
            graph LR
                TX[Transaction Data] --> CTX1[Transaction Context]
                CTX1 --> CTX2[Pattern Context]
                CTX1 --> CTX3[User Profile Context]
                CTX1 --> CTX4[Merchant Context]
                CTX2 --> CTX5[Investigation Context]
                CTX3 --> CTX5
                CTX4 --> CTX5
                CTX5 --> CTX6[Decision Context]
                CTX6 --> CTX7[Feedback Context]
                CTX7 --> CTX8[Learning Context]
            ```
            """)
            
            st.markdown("""
            The Model Context Protocol (MCP) enables the system to maintain context across different agents:
            
            1. Each agent creates a **context** for its analysis
            2. Contexts can contain **structured data**, **text**, and **references** to other contexts
            3. Contexts are **linked** to form a knowledge graph
            4. The MCP allows for **multi-modal** context handling (text, structured data, images)
            5. Agents can **merge contexts** to create a comprehensive view
            """)
            
            # Show example context
            with st.expander("Example MCP Context"):
                st.code("""
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
                """, language="json")
        
        with tabs[2]:
            st.markdown("### Agent-to-Agent (A2A) Communication")
            
            # Create a visualization of A2A communication
            st.markdown("""
            ```mermaid
            sequenceDiagram
                participant TA as Transaction Analysis Agent
                participant PD as Pattern Detection Agent
                participant IA as Investigation Agent
                participant DA as Decision Agent
                
                TA->>PD: transaction_analyzed
                Note over TA,PD: Transaction details and initial fraud score
                PD->>IA: patterns_detected
                Note over PD,IA: Detected patterns and updated score
                IA->>DA: investigation_completed
                Note over IA,DA: Investigation results and final score
                DA->>TA: decision_made
                Note over DA,TA: Final decision and reason
            ```
            """)
            
            st.markdown("""
            The Agent-to-Agent (A2A) communication system enables structured message passing between agents:
            
            1. Agents send **messages** with specific message types
            2. Messages contain **content** relevant to the task
            3. Messages can reference **contexts** from the MCP
            4. Agents can **subscribe** to receive messages from other agents
            5. The system supports both **direct messaging** and **broadcasting**
            """)
            
            # Show example message
            with st.expander("Example A2A Message"):
                st.code("""
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
                """, language="json")
    
    def render_adk_visualization(self):
        """Render visualization of Agent Development Kit (ADK) for Neo4j"""
        st.subheader("Agent Development Kit (ADK) for Neo4j")
        
        # Create tabs for different visualizations
        tabs = st.tabs(["Graph Schema", "ADK Capabilities", "Query Examples"])
        
        with tabs[0]:
            st.markdown("### Neo4j Graph Schema")
            
            # Display the graph schema
            st.markdown("""
            ```mermaid
            graph TD
                User[User] -->|OWNS| Card[Card]
                Card -->|MADE| Transaction[Transaction]
                Transaction -->|AT| Merchant[Merchant]
                Transaction -->|IN| Location[Location]
                Transaction -->|USING| Device[Device]
                Device -->|FROM| IPAddress[IPAddress]
                Transaction -->|FOLLOWED_BY| Transaction
                Transaction -->|SIMILAR_TO| FraudPattern[FraudPattern]
                User -->|FREQUENTLY_VISITS| Merchant
                User -->|TYPICALLY_IN| Location
                Transaction -->|TRIGGERED| Alert[Alert]
                Alert -->|FLAGGED| Transaction
            ```
            """)
            
            st.markdown("""
            The Neo4j graph database schema captures the relationships between entities:
            
            1. **Users** own **Cards** which make **Transactions**
            2. Transactions occur at **Merchants** and in **Locations**
            3. Online transactions use **Devices** with **IP Addresses**
            4. Transactions can be linked to **Fraud Patterns**
            5. The graph captures temporal relationships between transactions (**FOLLOWED_BY**)
            6. User behavior patterns are captured (**FREQUENTLY_VISITS**, **TYPICALLY_IN**)
            """)
        
        with tabs[1]:
            st.markdown("### ADK Capabilities")
            
            # List ADK capabilities
            st.markdown("""
            The Agent Development Kit (ADK) for Neo4j provides specialized capabilities for fraud detection:
            
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
            """)
        
        with tabs[2]:
            st.markdown("### Query Examples")
            
            # Show example queries
            with st.expander("Velocity Anomaly Detection"):
                st.code("""
MATCH (c:Card)-[:MADE]->(t1:Transaction)-[r:FOLLOWED_BY]->(t2:Transaction)
WHERE r.timeDifference <= 300  // 5 minutes
WITH c, t1, collect(t2) as subsequent_txs
WHERE size(subsequent_txs) >= 2
MATCH (u:User)-[:OWNS]->(c)
RETURN u.userId as user_id, c.cardId as card_id, 
       t1.transactionId as first_transaction_id,
       [tx in subsequent_txs | tx.transactionId] as subsequent_transaction_ids,
       size(subsequent_txs) + 1 as transaction_count
ORDER BY transaction_count DESC
                """, language="cypher")
            
            with st.expander("Location Anomaly Detection"):
                st.code("""
MATCH (c:Card)-[:MADE]->(t1:Transaction)-[:IN]->(l1:Location)
MATCH (c)-[:MADE]->(t2:Transaction)-[:IN]->(l2:Location)
WHERE t1.transactionId <> t2.transactionId
  AND duration.between(t1.timestamp, t2.timestamp).seconds <= 7200  // 2 hours
  AND point.distance(l1.point, l2.point) / 1000 >= 500  // 500 km
MATCH (u:User)-[:OWNS]->(c)
RETURN u.userId as user_id, c.cardId as card_id,
       t1.transactionId as transaction1_id, t2.transactionId as transaction2_id,
       l1.city as city1, l2.city as city2,
       l1.country as country1, l2.country as country2,
       round(point.distance(l1.point, l2.point) / 1000) as distance_km,
       duration.between(t1.timestamp, t2.timestamp).seconds as time_diff_seconds
ORDER BY distance_km DESC
                """, language="cypher")
            
            with st.expander("GraphRAG Knowledge Retrieval"):
                st.code("""
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
                """, language="cypher")
    
    def render_dashboard(self):
        """Render the complete dashboard"""
        # Simulate real-time data
        self.simulate_real_time_data()
        
        # Render header
        self.render_header()
        
        # Render metrics
        self.render_metrics()
        
        # Create tabs for different sections
        tabs = st.tabs(["Transaction Feed", "Fraud Analytics", "Multi-Agent System", "ADK Visualization"])
        
        with tabs[0]:
            # Render transaction feed
            self.render_transaction_feed()
            
            # Render transaction details if selected
            self.render_transaction_details()
        
        with tabs[1]:
            # Create columns for charts
            col1, col2 = st.columns(2)
            
            with col1:
                self.render_fraud_patterns_chart()
            
            with col2:
                self.render_fraud_merchants_chart()
            
            # Render fraud by location
            self.render_fraud_locations_chart()
        
        with tabs[2]:
            # Render multi-agent visualization
            self.render_multi_agent_visualization()
        
        with tabs[3]:
            # Render ADK visualization
            self.render_adk_visualization()

def main():
    parser = argparse.ArgumentParser(description='Real-time Transaction Monitoring Dashboard')
    parser.add_argument('--neo4j-uri', type=str, default='bolt://localhost:7687',
                        help='Neo4j URI')
    parser.add_argument('--neo4j-username', type=str, default='neo4j',
                        help='Neo4j username')
    parser.add_argument('--neo4j-password', type=str, default='password',
                        help='Neo4j password')
    parser.add_argument('--data-dir', type=str, default='/home/ubuntu/credit_card_fraud_detection/data',
                        help='Directory containing data CSV files')
    
    args = parser.parse_args()
    
    # Initialize dashboard
    dashboard = TransactionDashboard(
        neo4j_uri=args.neo4j_uri,
        neo4j_username=args.neo4j_username,
        neo4j_password=args.neo4j_password,
        data_dir=args.data_dir
    )
    
    # Render dashboard
    dashboard.render_dashboard()

if __name__ == "__main__":
    main()
