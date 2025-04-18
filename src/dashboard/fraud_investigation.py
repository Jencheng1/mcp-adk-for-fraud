#!/usr/bin/env python3
"""
Fraud Investigation Dashboard for Credit Card Fraud Detection

This script implements a Streamlit dashboard for investigating fraud patterns
and visualizing the multi-agent system with MCP, A2A, and ADK integration.
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
import networkx as nx
from datetime import datetime, timedelta
from neo4j import GraphDatabase

# Add parent directory to path for imports
import sys
sys.path.append('/home/ubuntu/credit_card_fraud_detection/src/neo4j')
from neo4j_adk import Neo4jADK

# Set page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection - Investigation Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dashboard styles
st.markdown("""
<style>
    .fraud-card {
        background-color: #ffcccb;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #ff0000;
        margin-bottom: 15px;
    }
    .investigation-card {
        background-color: #e6f3ff;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #0066cc;
        margin-bottom: 15px;
    }
    .pattern-card {
        background-color: #e6ffe6;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #00cc00;
        margin-bottom: 15px;
    }
    .agent-card {
        background-color: #fff2e6;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #ff9933;
        margin-bottom: 15px;
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

class FraudInvestigationDashboard:
    """Dashboard for investigating credit card fraud patterns and multi-agent system"""
    
    def __init__(self, neo4j_uri, neo4j_username, neo4j_password, data_dir):
        """Initialize the dashboard"""
        self.neo4j_adk = Neo4jADK(neo4j_uri, neo4j_username, neo4j_password)
        self.data_dir = data_dir
        
        # Load transaction data
        self.load_data()
        
        # Initialize session state
        if 'selected_transaction' not in st.session_state:
            st.session_state.selected_transaction = None
        if 'selected_pattern' not in st.session_state:
            st.session_state.selected_pattern = None
        if 'selected_user' not in st.session_state:
            st.session_state.selected_user = None
        if 'agent_logs' not in st.session_state:
            st.session_state.agent_logs = self.generate_sample_agent_logs()
        if 'mcp_contexts' not in st.session_state:
            st.session_state.mcp_contexts = self.generate_sample_mcp_contexts()
        if 'a2a_messages' not in st.session_state:
            st.session_state.a2a_messages = self.generate_sample_a2a_messages()
    
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
            
            # Filter fraudulent transactions
            self.fraud_df = self.transactions_df[self.transactions_df['is_fraudulent'] == True].copy()
            
            # Group by fraud pattern
            self.fraud_by_pattern = self.fraud_df.groupby('fraud_pattern').size().reset_index(name='count')
            self.fraud_by_pattern = self.fraud_by_pattern.sort_values('count', ascending=False)
            
            # Group by merchant
            self.fraud_by_merchant = self.fraud_df.groupby('merchant_id').size().reset_index(name='count')
            self.fraud_by_merchant['merchant_name'] = self.fraud_by_merchant['merchant_id'].map(self.merchant_names)
            self.fraud_by_merchant = self.fraud_by_merchant.sort_values('count', ascending=False)
            
            # Group by location
            self.fraud_by_location = self.fraud_df.groupby('location_id').size().reset_index(name='count')
            self.fraud_by_location['location_name'] = self.fraud_by_location['location_id'].map(self.location_names)
            self.fraud_by_location = self.fraud_by_location.sort_values('count', ascending=False)
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            self.transactions_df = pd.DataFrame()
            self.users_df = pd.DataFrame()
            self.cards_df = pd.DataFrame()
            self.merchants_df = pd.DataFrame()
            self.locations_df = pd.DataFrame()
            self.fraud_df = pd.DataFrame()
            self.fraud_by_pattern = pd.DataFrame()
            self.fraud_by_merchant = pd.DataFrame()
            self.fraud_by_location = pd.DataFrame()
    
    def generate_sample_agent_logs(self):
        """Generate sample agent logs for demonstration"""
        agent_types = [
            "transaction_analysis_agent",
            "pattern_detection_agent",
            "investigation_agent",
            "decision_agent",
            "user_profile_agent",
            "merchant_risk_agent",
            "feedback_collection_agent",
            "learning_agent"
        ]
        
        logs = []
        
        # Generate sample logs for each agent type
        for agent_type in agent_types:
            for i in range(5):  # 5 logs per agent
                timestamp = datetime.now() - timedelta(minutes=random.randint(0, 60))
                
                if agent_type == "transaction_analysis_agent":
                    action = random.choice([
                        "Analyzing transaction",
                        "Calculating initial fraud score",
                        "Detecting fraud indicators",
                        "Sending analysis results"
                    ])
                    details = {
                        "transaction_id": f"tx_{random.randint(10000, 99999)}",
                        "initial_fraud_score": round(random.random(), 2),
                        "fraud_indicators": random.sample(["high_amount", "unusual_merchant", "high_risk_device", "suspicious_ip"], k=random.randint(0, 3))
                    }
                
                elif agent_type == "pattern_detection_agent":
                    action = random.choice([
                        "Detecting patterns",
                        "Analyzing transaction patterns",
                        "Checking for velocity anomalies",
                        "Checking for location anomalies"
                    ])
                    details = {
                        "transaction_id": f"tx_{random.randint(10000, 99999)}",
                        "patterns_detected": random.sample(["velocity_anomaly", "location_anomaly", "unusual_merchant", "high_risk_device"], k=random.randint(0, 3)),
                        "pattern_score": round(random.random(), 2)
                    }
                
                elif agent_type == "investigation_agent":
                    action = random.choice([
                        "Investigating suspicious transaction",
                        "Analyzing transaction chain",
                        "Retrieving similar fraud cases",
                        "Calculating investigation score"
                    ])
                    details = {
                        "transaction_id": f"tx_{random.randint(10000, 99999)}",
                        "investigation_score_adjustment": round(random.random() * 0.3, 2),
                        "similar_cases_found": random.randint(0, 5)
                    }
                
                elif agent_type == "decision_agent":
                    action = random.choice([
                        "Making final determination",
                        "Calculating final fraud score",
                        "Determining fraud pattern",
                        "Creating fraud alert"
                    ])
                    decision = random.choice(["approve", "deny", "flag_for_review", "additional_authentication"])
                    details = {
                        "transaction_id": f"tx_{random.randint(10000, 99999)}",
                        "final_fraud_score": round(random.random(), 2),
                        "decision": decision,
                        "fraud_pattern": random.choice(["velocity_abuse", "location_anomaly", "card_not_present", "merchant_fraud", None]) if decision == "deny" else None
                    }
                
                else:
                    action = f"{agent_type} processing"
                    details = {
                        "transaction_id": f"tx_{random.randint(10000, 99999)}",
                        "processing_time": round(random.random() * 0.5, 3)
                    }
                
                logs.append({
                    "timestamp": timestamp,
                    "agent_type": agent_type,
                    "action": action,
                    "details": details
                })
        
        # Sort logs by timestamp (newest first)
        logs.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return logs
    
    def generate_sample_mcp_contexts(self):
        """Generate sample MCP contexts for demonstration"""
        context_types = [
            "transaction_analysis",
            "pattern_detection",
            "investigation",
            "decision",
            "user_profile",
            "merchant_risk",
            "feedback_collection",
            "learning"
        ]
        
        contexts = []
        
        # Generate 3 transaction IDs
        transaction_ids = [f"tx_{random.randint(10000, 99999)}" for _ in range(3)]
        
        # For each transaction, create contexts for each type
        for tx_id in transaction_ids:
            for context_type in context_types:
                timestamp = datetime.now() - timedelta(minutes=random.randint(0, 60))
                context_id = f"ctx_{context_type}_{tx_id[3:]}"
                
                # Create base context
                context = {
                    "context_id": context_id,
                    "context_type": context_type,
                    "created_at": timestamp.isoformat(),
                    "content": {},
                    "metadata": {
                        "agent": f"{context_type}_agent",
                        "processing_time": round(random.random() * 0.5, 3)
                    },
                    "references": []
                }
                
                # Add content based on context type
                if context_type == "transaction_analysis":
                    context["content"]["transaction"] = {
                        "transaction_id": tx_id,
                        "amount": round(random.uniform(10, 2000), 2),
                        "timestamp": timestamp.isoformat(),
                        "merchant_id": f"merch_{random.randint(1000, 9999)}",
                        "user_id": f"user_{random.randint(1000, 9999)}"
                    }
                    context["content"]["analysis_results"] = {
                        "initial_fraud_score": round(random.random(), 2),
                        "fraud_indicators": random.sample(["high_amount", "unusual_merchant", "high_risk_device", "suspicious_ip"], k=random.randint(0, 3))
                    }
                
                elif context_type == "pattern_detection":
                    context["content"]["pattern_results"] = {
                        "patterns_detected": random.sample(["velocity_anomaly", "location_anomaly", "unusual_merchant", "high_risk_device"], k=random.randint(0, 3)),
                        "pattern_score": round(random.random(), 2)
                    }
                    context["references"].append(f"ctx_transaction_analysis_{tx_id[3:]}")
                
                elif context_type == "investigation":
                    context["content"]["investigation_results"] = {
                        "transaction_chain": [tx_id] + [f"tx_{random.randint(10000, 99999)}" for _ in range(random.randint(1, 3))],
                        "similar_fraud_cases": random.randint(0, 5),
                        "score_adjustment": round(random.random() * 0.3, 2)
                    }
                    context["references"].append(f"ctx_transaction_analysis_{tx_id[3:]}")
                    context["references"].append(f"ctx_pattern_detection_{tx_id[3:]}")
                
                elif context_type == "decision":
                    decision = random.choice(["approve", "deny", "flag_for_review", "additional_authentication"])
                    context["content"]["decision_result"] = {
                        "decision": decision,
                        "reason": f"Based on fraud score and patterns",
                        "fraud_score": round(random.random(), 2),
                        "fraud_pattern": random.choice(["velocity_abuse", "location_anomaly", "card_not_present", "merchant_fraud", None]) if decision == "deny" else None
                    }
                    context["references"].append(f"ctx_transaction_analysis_{tx_id[3:]}")
                    context["references"].append(f"ctx_pattern_detection_{tx_id[3:]}")
                    if random.random() > 0.5:
                        context["references"].append(f"ctx_investigation_{tx_id[3:]}")
                
                else:
                    context["content"]["processing_result"] = {
                        "processed": True,
                        "processing_time": round(random.random() * 0.5, 3)
                    }
                    context["references"].append(f"ctx_transaction_analysis_{tx_id[3:]}")
                
                contexts.append(context)
        
        # Sort contexts by timestamp (newest first)
        contexts.sort(key=lambda x: x["created_at"], reverse=True)
        
        return contexts
    
    def generate_sample_a2a_messages(self):
        """Generate sample A2A messages for demonstration"""
        agent_types = [
            "transaction_analysis_agent",
            "pattern_detection_agent",
            "investigation_agent",
            "decision_agent",
            "user_profile_agent",
            "merchant_risk_agent",
            "feedback_collection_agent",
            "learning_agent"
        ]
        
        message_types = [
            "transaction_analyzed",
            "patterns_detected",
            "investigation_completed",
            "decision_made",
            "profile_analyzed",
            "merchant_analyzed",
            "feedback_collected",
            "model_updated"
        ]
        
        messages = []
        
        # Generate 3 transaction IDs
        transaction_ids = [f"tx_{random.randint(10000, 99999)}" for _ in range(3)]
        
        # For each transaction, create a sequence of messages
        for tx_id in transaction_ids:
            # Create context IDs for this transaction
            context_ids = {
                agent_type.replace("_agent", ""): f"ctx_{agent_type.replace('_agent', '')}_{tx_id[3:]}"
                for agent_type in agent_types
            }
            
            # Transaction analysis -> Pattern detection
            timestamp = datetime.now() - timedelta(minutes=random.randint(30, 60))
            messages.append({
                "message_id": f"msg_{len(messages) + 1:06d}",
                "sender": "transaction_analysis_agent",
                "receiver": "pattern_detection_agent",
                "message_type": "transaction_analyzed",
                "priority": "medium",
                "content": {
                    "transaction_id": tx_id,
                    "fraud_score": round(random.random(), 2),
                    "fraud_indicators": random.sample(["high_amount", "unusual_merchant", "high_risk_device", "suspicious_ip"], k=random.randint(0, 3))
                },
                "timestamp": timestamp.isoformat(),
                "context_references": [context_ids["transaction_analysis"]],
                "status": "read"
            })
            
            # Pattern detection -> Investigation or Decision
            timestamp = datetime.now() - timedelta(minutes=random.randint(20, 30))
            receiver = "investigation_agent" if random.random() > 0.5 else "decision_agent"
            messages.append({
                "message_id": f"msg_{len(messages) + 1:06d}",
                "sender": "pattern_detection_agent",
                "receiver": receiver,
                "message_type": "patterns_detected",
                "priority": "high" if receiver == "investigation_agent" else "medium",
                "content": {
                    "transaction_id": tx_id,
                    "patterns_detected": random.sample(["velocity_anomaly", "location_anomaly", "unusual_merchant", "high_risk_device"], k=random.randint(0, 3)),
                    "pattern_score": round(random.random(), 2),
                    "combined_score": round(random.random(), 2)
                },
                "timestamp": timestamp.isoformat(),
                "context_references": [context_ids["transaction_analysis"], context_ids["pattern_detection"]],
                "status": "read"
            })
            
            # If went to investigation, then investigation -> decision
            if receiver == "investigation_agent":
                timestamp = datetime.now() - timedelta(minutes=random.randint(10, 20))
                messages.append({
                    "message_id": f"msg_{len(messages) + 1:06d}",
                    "sender": "investigation_agent",
                    "receiver": "decision_agent",
                    "message_type": "investigation_completed",
                    "priority": "high",
                    "content": {
                        "transaction_id": tx_id,
                        "investigation_results": {
                            "transaction_chain": [tx_id] + [f"tx_{random.randint(10000, 99999)}" for _ in range(random.randint(1, 3))],
                            "similar_fraud_cases": random.randint(0, 5)
                        },
                        "score_adjustment": round(random.random() * 0.3, 2),
                        "final_score": round(random.random(), 2)
                    },
                    "timestamp": timestamp.isoformat(),
                    "context_references": [
                        context_ids["transaction_analysis"], 
                        context_ids["pattern_detection"],
                        context_ids["investigation"]
                    ],
                    "status": "read"
                })
            
            # Decision -> Feedback
            timestamp = datetime.now() - timedelta(minutes=random.randint(5, 10))
            decision = random.choice(["approve", "deny", "flag_for_review", "additional_authentication"])
            messages.append({
                "message_id": f"msg_{len(messages) + 1:06d}",
                "sender": "decision_agent",
                "receiver": "feedback_collection_agent",
                "message_type": "decision_made",
                "priority": "medium",
                "content": {
                    "decision": decision,
                    "reason": f"Based on fraud score and patterns",
                    "fraud_score": round(random.random(), 2),
                    "fraud_pattern": random.choice(["velocity_abuse", "location_anomaly", "card_not_present", "merchant_fraud", None]) if decision == "deny" else None
                },
                "timestamp": timestamp.isoformat(),
                "context_references": [context_ids["decision"]],
                "status": "read"
            })
            
            # Feedback -> Learning
            timestamp = datetime.now() - timedelta(minutes=random.randint(0, 5))
            messages.append({
                "message_id": f"msg_{len(messages) + 1:06d}",
                "sender": "feedback_collection_agent",
                "receiver": "learning_agent",
                "message_type": "feedback_collected",
                "priority": "low",
                "content": {
                    "transaction_id": tx_id,
                    "decision": decision,
                    "feedback_source": "simulated",
                    "is_correct": random.random() > 0.2,
                    "feedback_type": "true_positive" if decision == "deny" else "true_negative",
                    "confidence": round(random.uniform(0.7, 1.0), 2)
                },
                "timestamp": timestamp.isoformat(),
                "context_references": [context_ids["feedback_collection"]],
                "status": "sent"
            })
            
            # Add some random messages for user profile and merchant risk
            if random.random() > 0.5:
                timestamp = datetime.now() - timedelta(minutes=random.randint(40, 50))
                messages.append({
                    "message_id": f"msg_{len(messages) + 1:06d}",
                    "sender": "user_profile_agent",
                    "receiver": "transaction_analysis_agent",
                    "message_type": "profile_analyzed",
                    "priority": "medium",
                    "content": {
                        "user_id": f"user_{random.randint(1000, 9999)}",
                        "transaction_id": tx_id,
                        "profile_analysis": {
                            "amount_ratio": round(random.uniform(0.5, 5.0), 2),
                            "is_frequent_merchant": random.choice([True, False]),
                            "is_typical_location": random.choice([True, False])
                        },
                        "risk_adjustment": round(random.random() * 0.3, 2)
                    },
                    "timestamp": timestamp.isoformat(),
                    "context_references": [context_ids["user_profile"]],
                    "status": "read"
                })
            
            if random.random() > 0.5:
                timestamp = datetime.now() - timedelta(minutes=random.randint(40, 50))
                messages.append({
                    "message_id": f"msg_{len(messages) + 1:06d}",
                    "sender": "merchant_risk_agent",
                    "receiver": "transaction_analysis_agent",
                    "message_type": "merchant_analyzed",
                    "priority": "medium",
                    "content": {
                        "merchant_id": f"merch_{random.randint(1000, 9999)}",
                        "transaction_id": tx_id,
                        "risk_analysis": {
                            "merchant_risk_score": round(random.random(), 2),
                            "is_high_risk": random.choice([True, False]),
                            "fraud_rate": round(random.random() * 0.05, 3)
                        },
                        "risk_adjustment": round(random.random() * 0.3, 2)
                    },
                    "timestamp": timestamp.isoformat(),
                    "context_references": [context_ids["merchant_risk"]],
                    "status": "read"
                })
        
        # Sort messages by timestamp (newest first)
        messages.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return messages
    
    def render_header(self):
        """Render the dashboard header"""
        st.title("Credit Card Fraud Detection - Investigation Dashboard")
        st.markdown("Investigate fraud patterns and visualize the multi-agent system with MCP, A2A, and ADK integration")
        
        # Add refresh button
        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("Refresh Data"):
                st.experimental_rerun()
    
    def render_fraud_patterns_overview(self):
        """Render overview of fraud patterns"""
        st.subheader("Fraud Patterns Overview")
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_transactions = len(self.transactions_df)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_transactions:,}</div>
                <div class="metric-label">Total Transactions</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_fraud = len(self.fraud_df)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_fraud:,}</div>
                <div class="metric-label">Fraudulent Transactions</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            fraud_rate = (total_fraud / total_transactions * 100) if total_transactions > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{fraud_rate:.2f}%</div>
                <div class="metric-label">Fraud Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            unique_patterns = self.fraud_df['fraud_pattern'].nunique()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{unique_patterns}</div>
                <div class="metric-label">Unique Fraud Patterns</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Create columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Fraud by pattern chart
            if not self.fraud_by_pattern.empty:
                fig = px.bar(
                    self.fraud_by_pattern,
                    x='fraud_pattern',
                    y='count',
                    color='fraud_pattern',
                    title='Fraud by Pattern'
                )
                
                fig.update_layout(
                    xaxis_title='Fraud Pattern',
                    yaxis_title='Number of Fraudulent Transactions',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No fraud pattern data available")
        
        with col2:
            # Fraud by merchant chart
            if not self.fraud_by_merchant.empty:
                # Take top 10 merchants
                top_merchants = self.fraud_by_merchant.head(10)
                
                fig = px.bar(
                    top_merchants,
                    x='merchant_name',
                    y='count',
                    color='merchant_name',
                    title='Top 10 Merchants with Fraud'
                )
                
                fig.update_layout(
                    xaxis_title='Merchant',
                    yaxis_title='Number of Fraudulent Transactions',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No fraud merchant data available")
    
    def render_pattern_details(self):
        """Render details for a selected fraud pattern"""
        st.subheader("Fraud Pattern Details")
        
        # Get unique patterns
        patterns = self.fraud_df['fraud_pattern'].unique()
        
        # Create a selectbox for patterns
        selected_pattern = st.selectbox(
            "Select a fraud pattern to investigate",
            options=patterns,
            index=0 if len(patterns) > 0 else None
        )
        
        if selected_pattern:
            # Filter transactions by pattern
            pattern_transactions = self.fraud_df[self.fraud_df['fraud_pattern'] == selected_pattern].copy()
            
            # Add merchant and location names
            pattern_transactions['merchant_name'] = pattern_transactions['merchant_id'].map(self.merchant_names)
            pattern_transactions['location_name'] = pattern_transactions['location_id'].map(self.location_names)
            pattern_transactions['user_name'] = pattern_transactions['user_id'].map(self.user_names)
            
            # Display pattern information
            st.markdown(f"""
            <div class="pattern-card">
                <h3>{selected_pattern}</h3>
                <p><strong>Number of transactions:</strong> {len(pattern_transactions)}</p>
                <p><strong>Average amount:</strong> ${pattern_transactions['amount'].mean():.2f}</p>
                <p><strong>Average fraud score:</strong> {pattern_transactions['fraud_score'].mean():.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create tabs for different views
            tabs = st.tabs(["Transactions", "Merchants", "Locations", "Time Analysis"])
            
            with tabs[0]:
                # Display transactions table
                st.dataframe(
                    pattern_transactions[[
                        'transaction_id', 'timestamp', 'amount', 'merchant_name', 
                        'location_name', 'user_name', 'fraud_score'
                    ]].sort_values('timestamp', ascending=False),
                    use_container_width=True
                )
            
            with tabs[1]:
                # Group by merchant
                merchant_counts = pattern_transactions.groupby('merchant_name').size().reset_index(name='count')
                merchant_counts = merchant_counts.sort_values('count', ascending=False)
                
                # Create bar chart
                fig = px.bar(
                    merchant_counts,
                    x='merchant_name',
                    y='count',
                    color='merchant_name',
                    title=f'Merchants with {selected_pattern} Fraud'
                )
                
                fig.update_layout(
                    xaxis_title='Merchant',
                    yaxis_title='Number of Fraudulent Transactions',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tabs[2]:
                # Group by location
                location_counts = pattern_transactions.groupby('location_name').size().reset_index(name='count')
                location_counts = location_counts.sort_values('count', ascending=False)
                
                # Create bar chart
                fig = px.bar(
                    location_counts,
                    x='location_name',
                    y='count',
                    color='location_name',
                    title=f'Locations with {selected_pattern} Fraud'
                )
                
                fig.update_layout(
                    xaxis_title='Location',
                    yaxis_title='Number of Fraudulent Transactions',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tabs[3]:
                # Group by hour of day
                pattern_transactions['hour'] = pattern_transactions['timestamp'].dt.hour
                hour_counts = pattern_transactions.groupby('hour').size().reset_index(name='count')
                
                # Create line chart
                fig = px.line(
                    hour_counts,
                    x='hour',
                    y='count',
                    title=f'Time Distribution of {selected_pattern} Fraud'
                )
                
                fig.update_layout(
                    xaxis_title='Hour of Day',
                    yaxis_title='Number of Fraudulent Transactions',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def render_transaction_investigation(self):
        """Render transaction investigation interface"""
        st.subheader("Transaction Investigation")
        
        # Create a text input for transaction ID
        transaction_id = st.text_input("Enter a transaction ID to investigate", "")
        
        if transaction_id:
            # Try to find the transaction in the dataset
            transaction = self.transactions_df[self.transactions_df['transaction_id'] == transaction_id]
            
            if not transaction.empty:
                # Get the transaction details
                tx = transaction.iloc[0]
                
                # Get related entities
                user_id = tx['user_id']
                card_id = tx['card_id']
                merchant_id = tx['merchant_id']
                location_id = tx['location_id']
                
                user = self.users_df[self.users_df['user_id'] == user_id].iloc[0] if user_id in self.users_df['user_id'].values else None
                card = self.cards_df[self.cards_df['card_id'] == card_id].iloc[0] if card_id in self.cards_df['card_id'].values else None
                merchant = self.merchants_df[self.merchants_df['merchant_id'] == merchant_id].iloc[0] if merchant_id in self.merchants_df['merchant_id'].values else None
                location = self.locations_df[self.locations_df['location_id'] == location_id].iloc[0] if location_id in self.locations_df['location_id'].values else None
                
                # Display transaction information
                st.markdown(f"""
                <div class="{'fraud-card' if tx['is_fraudulent'] else 'investigation-card'}">
                    <h3>Transaction {tx['transaction_id']}</h3>
                    <p><strong>Timestamp:</strong> {tx['timestamp']}</p>
                    <p><strong>Amount:</strong> ${tx['amount']:.2f}</p>
                    <p><strong>Fraud Score:</strong> {tx['fraud_score']:.2f}</p>
                    <p><strong>Fraudulent:</strong> {'Yes' if tx['is_fraudulent'] else 'No'}</p>
                    {f"<p><strong>Fraud Pattern:</strong> {tx['fraud_pattern']}</p>" if tx['is_fraudulent'] and pd.notna(tx['fraud_pattern']) else ""}
                </div>
                """, unsafe_allow_html=True)
                
                # Create tabs for different entity views
                tabs = st.tabs(["User", "Card", "Merchant", "Location", "Graph View"])
                
                with tabs[0]:
                    if user is not None:
                        st.markdown(f"""
                        <div class="investigation-card">
                            <h3>User {user['user_id']}</h3>
                            <p><strong>Name:</strong> {user['name']}</p>
                            <p><strong>Email:</strong> {user['email']}</p>
                            <p><strong>Risk Score:</strong> {user['risk_score']:.2f}</p>
                            <p><strong>Fraud History:</strong> {'Yes' if user['fraud_history'] else 'No'}</p>
                            <p><strong>Account Creation:</strong> {user['account_creation_date']}</p>
                            <p><strong>Segment:</strong> {user['segment']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Get user's transactions
                        user_transactions = self.transactions_df[self.transactions_df['user_id'] == user_id].copy()
                        user_transactions['merchant_name'] = user_transactions['merchant_id'].map(self.merchant_names)
                        
                        # Display user's recent transactions
                        st.markdown("### Recent Transactions")
                        st.dataframe(
                            user_transactions[[
                                'transaction_id', 'timestamp', 'amount', 'merchant_name', 
                                'is_fraudulent', 'fraud_score'
                            ]].sort_values('timestamp', ascending=False).head(10),
                            use_container_width=True
                        )
                    else:
                        st.info("User information not available")
                
                with tabs[1]:
                    if card is not None:
                        st.markdown(f"""
                        <div class="investigation-card">
                            <h3>Card {card['card_id']}</h3>
                            <p><strong>Card Type:</strong> {card['card_type']}</p>
                            <p><strong>Last Four Digits:</strong> {card['last_four_digits']}</p>
                            <p><strong>Issue Date:</strong> {card['issue_date']}</p>
                            <p><strong>Expiry Date:</strong> {card['expiry_date']}</p>
                            <p><strong>Active:</strong> {'Yes' if card['is_active'] else 'No'}</p>
                            <p><strong>Blocked:</strong> {'Yes' if card['is_blocked'] else 'No'}</p>
                            <p><strong>Credit Limit:</strong> ${card['credit_limit']:.2f}</p>
                            <p><strong>Available Credit:</strong> ${card['available_credit']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Get card's transactions
                        card_transactions = self.transactions_df[self.transactions_df['card_id'] == card_id].copy()
                        card_transactions['merchant_name'] = card_transactions['merchant_id'].map(self.merchant_names)
                        
                        # Display card's recent transactions
                        st.markdown("### Recent Transactions")
                        st.dataframe(
                            card_transactions[[
                                'transaction_id', 'timestamp', 'amount', 'merchant_name', 
                                'is_fraudulent', 'fraud_score'
                            ]].sort_values('timestamp', ascending=False).head(10),
                            use_container_width=True
                        )
                    else:
                        st.info("Card information not available")
                
                with tabs[2]:
                    if merchant is not None:
                        st.markdown(f"""
                        <div class="investigation-card">
                            <h3>Merchant {merchant['merchant_id']}</h3>
                            <p><strong>Name:</strong> {merchant['name']}</p>
                            <p><strong>Category:</strong> {merchant['category']}</p>
                            <p><strong>MCC:</strong> {merchant['mcc']}</p>
                            <p><strong>Country:</strong> {merchant['country']}</p>
                            <p><strong>Risk Score:</strong> {merchant['risk_score']:.2f}</p>
                            <p><strong>Fraud Rate:</strong> {merchant['fraud_rate']:.4f}</p>
                            <p><strong>Average Transaction:</strong> ${merchant['avg_transaction_amount']:.2f}</p>
                            <p><strong>High Risk:</strong> {'Yes' if merchant['is_high_risk'] else 'No'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Get merchant's transactions
                        merchant_transactions = self.transactions_df[self.transactions_df['merchant_id'] == merchant_id].copy()
                        merchant_transactions['user_name'] = merchant_transactions['user_id'].map(self.user_names)
                        
                        # Display merchant's recent transactions
                        st.markdown("### Recent Transactions")
                        st.dataframe(
                            merchant_transactions[[
                                'transaction_id', 'timestamp', 'amount', 'user_name', 
                                'is_fraudulent', 'fraud_score'
                            ]].sort_values('timestamp', ascending=False).head(10),
                            use_container_width=True
                        )
                    else:
                        st.info("Merchant information not available")
                
                with tabs[3]:
                    if location is not None:
                        st.markdown(f"""
                        <div class="investigation-card">
                            <h3>Location {location['location_id']}</h3>
                            <p><strong>Country:</strong> {location['country']}</p>
                            <p><strong>City:</strong> {location['city']}</p>
                            <p><strong>Postal Code:</strong> {location['postal_code']}</p>
                            <p><strong>Address:</strong> {location['address']}</p>
                            <p><strong>Latitude:</strong> {location['latitude']}</p>
                            <p><strong>Longitude:</strong> {location['longitude']}</p>
                            <p><strong>Risk Score:</strong> {location['risk_score']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Get location's transactions
                        location_transactions = self.transactions_df[self.transactions_df['location_id'] == location_id].copy()
                        location_transactions['user_name'] = location_transactions['user_id'].map(self.user_names)
                        location_transactions['merchant_name'] = location_transactions['merchant_id'].map(self.merchant_names)
                        
                        # Display location's recent transactions
                        st.markdown("### Recent Transactions")
                        st.dataframe(
                            location_transactions[[
                                'transaction_id', 'timestamp', 'amount', 'user_name', 
                                'merchant_name', 'is_fraudulent', 'fraud_score'
                            ]].sort_values('timestamp', ascending=False).head(10),
                            use_container_width=True
                        )
                    else:
                        st.info("Location information not available")
                
                with tabs[4]:
                    st.markdown("### Transaction Graph")
                    
                    # Create a graph visualization
                    G = nx.DiGraph()
                    
                    # Add nodes
                    G.add_node(f"tx_{tx['transaction_id']}", type="transaction", label=f"Transaction\n${tx['amount']:.2f}")
                    
                    if user is not None:
                        G.add_node(f"user_{user['user_id']}", type="user", label=f"User\n{user['name']}")
                        G.add_edge(f"user_{user['user_id']}", f"tx_{tx['transaction_id']}", label="MADE")
                    
                    if card is not None:
                        G.add_node(f"card_{card['card_id']}", type="card", label=f"Card\n{card['last_four_digits']}")
                        G.add_edge(f"user_{user['user_id']}", f"card_{card['card_id']}", label="OWNS")
                        G.add_edge(f"card_{card['card_id']}", f"tx_{tx['transaction_id']}", label="USED_IN")
                    
                    if merchant is not None:
                        G.add_node(f"merchant_{merchant['merchant_id']}", type="merchant", label=f"Merchant\n{merchant['name']}")
                        G.add_edge(f"tx_{tx['transaction_id']}", f"merchant_{merchant['merchant_id']}", label="AT")
                    
                    if location is not None:
                        G.add_node(f"location_{location['location_id']}", type="location", label=f"Location\n{location['city']}, {location['country']}")
                        G.add_edge(f"tx_{tx['transaction_id']}", f"location_{location['location_id']}", label="IN")
                    
                    # Add fraud pattern if fraudulent
                    if tx['is_fraudulent'] and pd.notna(tx['fraud_pattern']):
                        G.add_node(f"pattern_{tx['fraud_pattern']}", type="pattern", label=f"Pattern\n{tx['fraud_pattern']}")
                        G.add_edge(f"tx_{tx['transaction_id']}", f"pattern_{tx['fraud_pattern']}", label="MATCHES")
                    
                    # Add related transactions (simplified for demo)
                    related_txs = self.transactions_df[
                        (self.transactions_df['card_id'] == card_id) & 
                        (self.transactions_df['transaction_id'] != tx['transaction_id'])
                    ].sort_values('timestamp').tail(3)
                    
                    for _, related_tx in related_txs.iterrows():
                        G.add_node(f"tx_{related_tx['transaction_id']}", type="transaction", label=f"Transaction\n${related_tx['amount']:.2f}")
                        G.add_edge(f"card_{card['card_id']}", f"tx_{related_tx['transaction_id']}", label="USED_IN")
                        
                        if related_tx['timestamp'] < tx['timestamp']:
                            G.add_edge(f"tx_{related_tx['transaction_id']}", f"tx_{tx['transaction_id']}", label="FOLLOWED_BY")
                        else:
                            G.add_edge(f"tx_{tx['transaction_id']}", f"tx_{related_tx['transaction_id']}", label="FOLLOWED_BY")
                    
                    # Create a Plotly figure
                    pos = nx.spring_layout(G, seed=42)
                    
                    edge_x = []
                    edge_y = []
                    edge_text = []
                    
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                        edge_text.append(G.edges[edge[0], edge[1]]['label'])
                    
                    edge_trace = go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=1, color='#888'),
                        hoverinfo='none',
                        mode='lines')
                    
                    node_x = []
                    node_y = []
                    node_text = []
                    node_color = []
                    
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        node_text.append(G.nodes[node]['label'])
                        
                        # Set color based on node type
                        if G.nodes[node]['type'] == 'transaction':
                            if 'is_fraudulent' in tx and tx['is_fraudulent']:
                                node_color.append('red')
                            else:
                                node_color.append('blue')
                        elif G.nodes[node]['type'] == 'user':
                            node_color.append('green')
                        elif G.nodes[node]['type'] == 'card':
                            node_color.append('purple')
                        elif G.nodes[node]['type'] == 'merchant':
                            node_color.append('orange')
                        elif G.nodes[node]['type'] == 'location':
                            node_color.append('brown')
                        elif G.nodes[node]['type'] == 'pattern':
                            node_color.append('red')
                        else:
                            node_color.append('gray')
                    
                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        hoverinfo='text',
                        text=node_text,
                        textposition="top center",
                        marker=dict(
                            showscale=False,
                            color=node_color,
                            size=15,
                            line_width=2))
                    
                    fig = go.Figure(data=[edge_trace, node_trace],
                                  layout=go.Layout(
                                      showlegend=False,
                                      hovermode='closest',
                                      margin=dict(b=20, l=5, r=5, t=40),
                                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                                  )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Transaction {transaction_id} not found")
    
    def render_agent_logs(self):
        """Render agent logs"""
        st.subheader("Agent Logs")
        
        # Create a filter for agent type
        agent_types = [
            "All Agents",
            "transaction_analysis_agent",
            "pattern_detection_agent",
            "investigation_agent",
            "decision_agent",
            "user_profile_agent",
            "merchant_risk_agent",
            "feedback_collection_agent",
            "learning_agent"
        ]
        
        selected_agent = st.selectbox("Filter by agent", agent_types)
        
        # Filter logs by agent type
        if selected_agent == "All Agents":
            filtered_logs = st.session_state.agent_logs
        else:
            filtered_logs = [log for log in st.session_state.agent_logs if log["agent_type"] == selected_agent]
        
        # Display logs
        for log in filtered_logs[:20]:  # Show only the most recent 20 logs
            st.markdown(f"""
            <div class="agent-card">
                <p><strong>{log['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</strong> - <em>{log['agent_type']}</em></p>
                <p><strong>Action:</strong> {log['action']}</p>
                <p><strong>Details:</strong></p>
                <pre>{json.dumps(log['details'], indent=2)}</pre>
            </div>
            """, unsafe_allow_html=True)
    
    def render_mcp_visualization(self):
        """Render MCP visualization"""
        st.subheader("Model Context Protocol (MCP) Visualization")
        
        # Create a filter for context type
        context_types = [
            "All Contexts",
            "transaction_analysis",
            "pattern_detection",
            "investigation",
            "decision",
            "user_profile",
            "merchant_risk",
            "feedback_collection",
            "learning"
        ]
        
        selected_context_type = st.selectbox("Filter by context type", context_types)
        
        # Filter contexts by type
        if selected_context_type == "All Contexts":
            filtered_contexts = st.session_state.mcp_contexts
        else:
            filtered_contexts = [ctx for ctx in st.session_state.mcp_contexts if ctx["context_type"] == selected_context_type]
        
        # Group contexts by transaction
        transaction_contexts = {}
        for ctx in filtered_contexts:
            # Extract transaction ID from context ID
            if "_tx_" in ctx["context_id"]:
                tx_id = "tx_" + ctx["context_id"].split("_")[-1]
                if tx_id not in transaction_contexts:
                    transaction_contexts[tx_id] = []
                transaction_contexts[tx_id].append(ctx)
        
        # Create tabs for each transaction
        if transaction_contexts:
            tabs = st.tabs(list(transaction_contexts.keys()))
            
            for i, (tx_id, contexts) in enumerate(transaction_contexts.items()):
                with tabs[i]:
                    # Create a graph visualization of contexts
                    G = nx.DiGraph()
                    
                    # Add nodes for each context
                    for ctx in contexts:
                        G.add_node(ctx["context_id"], type=ctx["context_type"], label=f"{ctx['context_type']}\n{ctx['context_id']}")
                    
                    # Add edges for references
                    for ctx in contexts:
                        for ref in ctx["references"]:
                            if ref in [c["context_id"] for c in contexts]:
                                G.add_edge(ref, ctx["context_id"])
                    
                    # Create a Plotly figure
                    pos = nx.spring_layout(G, seed=42)
                    
                    edge_x = []
                    edge_y = []
                    
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                    
                    edge_trace = go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=1, color='#888'),
                        hoverinfo='none',
                        mode='lines')
                    
                    node_x = []
                    node_y = []
                    node_text = []
                    node_color = []
                    
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        node_text.append(G.nodes[node]['label'])
                        
                        # Set color based on context type
                        if G.nodes[node]['type'] == 'transaction_analysis':
                            node_color.append('blue')
                        elif G.nodes[node]['type'] == 'pattern_detection':
                            node_color.append('green')
                        elif G.nodes[node]['type'] == 'investigation':
                            node_color.append('red')
                        elif G.nodes[node]['type'] == 'decision':
                            node_color.append('purple')
                        elif G.nodes[node]['type'] == 'user_profile':
                            node_color.append('orange')
                        elif G.nodes[node]['type'] == 'merchant_risk':
                            node_color.append('brown')
                        else:
                            node_color.append('gray')
                    
                    node_trace = go.Scatter(
                        x=node_x, y=edge_y,
                        mode='markers+text',
                        hoverinfo='text',
                        text=node_text,
                        textposition="top center",
                        marker=dict(
                            showscale=False,
                            color=node_color,
                            size=15,
                            line_width=2))
                    
                    fig = go.Figure(data=[edge_trace, node_trace],
                                  layout=go.Layout(
                                      title=f"Context Flow for {tx_id}",
                                      showlegend=False,
                                      hovermode='closest',
                                      margin=dict(b=20, l=5, r=5, t=40),
                                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                                  )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display context details
                    for ctx in contexts:
                        with st.expander(f"{ctx['context_type']} - {ctx['context_id']}"):
                            st.json(ctx)
        else:
            st.info("No contexts found for the selected filter")
    
    def render_a2a_visualization(self):
        """Render A2A visualization"""
        st.subheader("Agent-to-Agent (A2A) Communication Visualization")
        
        # Create a filter for message type
        message_types = [
            "All Messages",
            "transaction_analyzed",
            "patterns_detected",
            "investigation_completed",
            "decision_made",
            "profile_analyzed",
            "merchant_analyzed",
            "feedback_collected",
            "model_updated"
        ]
        
        selected_message_type = st.selectbox("Filter by message type", message_types)
        
        # Filter messages by type
        if selected_message_type == "All Messages":
            filtered_messages = st.session_state.a2a_messages
        else:
            filtered_messages = [msg for msg in st.session_state.a2a_messages if msg["message_type"] == selected_message_type]
        
        # Group messages by transaction
        transaction_messages = {}
        for msg in filtered_messages:
            # Extract transaction ID from content
            if "transaction_id" in msg["content"]:
                tx_id = msg["content"]["transaction_id"]
                if tx_id not in transaction_messages:
                    transaction_messages[tx_id] = []
                transaction_messages[tx_id].append(msg)
        
        # Create tabs for each transaction
        if transaction_messages:
            tabs = st.tabs(list(transaction_messages.keys()))
            
            for i, (tx_id, messages) in enumerate(transaction_messages.items()):
                with tabs[i]:
                    # Sort messages by timestamp
                    messages.sort(key=lambda x: x["timestamp"])
                    
                    # Create a sequence diagram visualization
                    st.markdown("""
                    ```mermaid
                    sequenceDiagram
                    """)
                    
                    # Add participants
                    agents = set()
                    for msg in messages:
                        agents.add(msg["sender"])
                        agents.add(msg["receiver"])
                    
                    for agent in agents:
                        st.markdown(f"    participant {agent.split('_')[0]} as {agent}")
                    
                    # Add messages
                    for msg in messages:
                        sender = msg["sender"]
                        receiver = msg["receiver"]
                        message_type = msg["message_type"]
                        
                        st.markdown(f"    {sender.split('_')[0]}->>+{receiver.split('_')[0]}: {message_type}")
                        
                        # Add note if needed
                        if "content" in msg and len(msg["content"]) > 0:
                            content_str = ", ".join([f"{k}: {v}" for k, v in msg["content"].items() if k != "transaction_id" and not isinstance(v, dict) and not isinstance(v, list)])
                            if content_str:
                                st.markdown(f"    Note over {sender.split('_')[0]},{receiver.split('_')[0]}: {content_str}")
                    
                    st.markdown("```")
                    
                    # Display message details
                    for msg in messages:
                        with st.expander(f"{msg['message_type']} - {msg['sender']} to {msg['receiver']}"):
                            st.json(msg)
        else:
            st.info("No messages found for the selected filter")
    
    def render_adk_visualization(self):
        """Render ADK visualization"""
        st.subheader("Agent Development Kit (ADK) for Neo4j")
        
        # Create tabs for different ADK features
        tabs = st.tabs(["Transaction Analysis", "Pattern Detection", "Investigation", "GraphRAG"])
        
        with tabs[0]:
            st.markdown("""
            ### Transaction Analysis
            
            The ADK provides specialized functions for analyzing individual transactions:
            
            ```python
            # Get transaction details
            tx_details = neo4j_adk.get_transaction_by_id("tx_123456")
            
            # Get user's recent transactions
            user_txs = neo4j_adk.get_user_transactions("user_789", limit=10)
            
            # Get card's recent transactions
            card_txs = neo4j_adk.get_card_transactions("card_456", limit=10)
            ```
            
            These functions leverage Neo4j's graph capabilities to retrieve not just the transaction data,
            but also all connected entities like the user, card, merchant, location, and device information.
            """)
            
            # Show example query result
            with st.expander("Example Transaction Analysis Result"):
                st.code("""
{
    "transaction": {
        "transactionId": "tx_123456",
        "timestamp": "2025-04-18T12:34:56.789Z",
        "amount": 1299.99,
        "currency": "USD",
        "status": "approved",
        "transactionType": "purchase",
        "paymentMethod": "credit",
        "isOnline": true,
        "isFraudulent": true,
        "fraudScore": 0.85,
        "fraudPattern": "card_not_present"
    },
    "card": {
        "cardId": "card_456",
        "cardType": "visa",
        "lastFourDigits": "1234",
        "isActive": true,
        "isBlocked": false
    },
    "merchant": {
        "merchantId": "merch_789",
        "name": "Premium Electronics",
        "category": "electronics",
        "riskScore": 0.3
    },
    "location": {
        "locationId": "loc_101",
        "country": "USA",
        "city": "New York",
        "riskScore": 0.2
    },
    "device": {
        "deviceId": "dev_202",
        "deviceType": "mobile",
        "browser": "Chrome",
        "operatingSystem": "iOS",
        "isKnownDevice": false,
        "riskScore": 0.7
    },
    "ip_address": {
        "ip": "203.0.113.1",
        "country": "Russia",
        "isProxy": true,
        "isVpn": false,
        "isTor": false,
        "riskScore": 0.8
    },
    "user": {
        "userId": "user_789",
        "name": "John Smith",
        "email": "john.smith@example.com",
        "riskScore": 0.4,
        "fraudHistory": false
    }
}
                """, language="json")
        
        with tabs[1]:
            st.markdown("""
            ### Pattern Detection
            
            The ADK provides specialized functions for detecting fraud patterns across multiple transactions:
            
            ```python
            # Detect velocity anomalies (multiple transactions in short time)
            velocity_anomalies = neo4j_adk.detect_velocity_anomalies(
                time_window_seconds=300,  # 5 minutes
                min_transactions=3
            )
            
            # Detect location anomalies (transactions in distant locations in short time)
            location_anomalies = neo4j_adk.detect_location_anomalies(
                max_time_diff_seconds=7200,  # 2 hours
                min_distance_km=500
            )
            
            # Detect unusual merchant activity
            unusual_merchants = neo4j_adk.detect_unusual_merchant_activity()
            
            # Detect transactions from high-risk devices
            high_risk_devices = neo4j_adk.detect_high_risk_devices()
            ```
            
            These functions leverage Neo4j's graph capabilities to detect complex patterns that would be
            difficult to identify with traditional SQL queries.
            """)
            
            # Show example query result
            with st.expander("Example Pattern Detection Result"):
                st.code("""
[
    {
        "user_id": "user_123",
        "card_id": "card_456",
        "first_transaction_id": "tx_789",
        "subsequent_transaction_ids": ["tx_790", "tx_791", "tx_792"],
        "transaction_count": 4,
        "time_span_seconds": 240
    },
    {
        "user_id": "user_456",
        "card_id": "card_789",
        "first_transaction_id": "tx_901",
        "subsequent_transaction_ids": ["tx_902", "tx_903"],
        "transaction_count": 3,
        "time_span_seconds": 180
    }
]
                """, language="json")
        
        with tabs[2]:
            st.markdown("""
            ### Investigation
            
            The ADK provides specialized functions for investigating suspicious transactions:
            
            ```python
            # Get the chain of transactions before and after a given transaction
            transaction_chain = neo4j_adk.get_transaction_chain(
                transaction_id="tx_123456",
                depth=3
            )
            
            # Get similar fraud patterns for a transaction
            similar_patterns = neo4j_adk.get_similar_fraud_patterns("tx_123456")
            
            # Calculate fraud risk for a user
            user_risk = neo4j_adk.get_user_fraud_risk("user_789")
            ```
            
            These functions help investigators understand the context around suspicious transactions
            and make more informed decisions.
            """)
            
            # Show example query result
            with st.expander("Example Investigation Result"):
                st.code("""
{
    "transaction_chain": [
        {
            "transaction_id": "tx_123",
            "timestamp": "2025-04-18T12:30:00.000Z",
            "amount": 25.99,
            "is_fraudulent": false,
            "card_id": "card_456",
            "merchant_name": "Coffee Shop",
            "city": "New York",
            "country": "USA"
        },
        {
            "transaction_id": "tx_124",
            "timestamp": "2025-04-18T12:45:00.000Z",
            "amount": 1299.99,
            "is_fraudulent": true,
            "card_id": "card_456",
            "merchant_name": "Premium Electronics",
            "city": "New York",
            "country": "USA"
        },
        {
            "transaction_id": "tx_125",
            "timestamp": "2025-04-18T12:50:00.000Z",
            "amount": 899.99,
            "is_fraudulent": true,
            "card_id": "card_456",
            "merchant_name": "Luxury Watches",
            "city": "New York",
            "country": "USA"
        }
    ],
    "similar_fraud_patterns": [
        {
            "pattern_id": "pattern_001",
            "description": "Multiple high-value purchases in short time",
            "pattern_type": "velocity_abuse",
            "similarity_score": 0.85
        },
        {
            "pattern_id": "pattern_004",
            "description": "Online transactions with suspicious patterns",
            "pattern_type": "card_not_present",
            "similarity_score": 0.75
        }
    ],
    "user_risk": {
        "user_id": "user_789",
        "name": "John Smith",
        "email": "john.smith@example.com",
        "fraud_count": 2,
        "high_risk_devices": 1,
        "suspicious_locations": 0,
        "base_risk_score": 0.4,
        "calculated_risk_score": 0.7
    }
}
                """, language="json")
        
        with tabs[3]:
            st.markdown("""
            ### GraphRAG Knowledge Retrieval
            
            The ADK provides a GraphRAG (Graph Retrieval Augmented Generation) capability for retrieving
            relevant knowledge for fraud investigation:
            
            ```python
            # Retrieve relevant knowledge for fraud investigation
            knowledge = neo4j_adk.retrieve_fraud_knowledge({
                "amount": 1299.99,
                "is_online": true,
                "merchant_category": "electronics"
            })
            ```
            
            GraphRAG combines the power of graph databases with retrieval-augmented generation to provide
            contextually relevant information for fraud investigation. It goes beyond simple keyword matching
            by leveraging the graph structure to find similar fraud cases based on transaction properties,
            connected entities, and fraud patterns.
            """)
            
            # Show example query result
            with st.expander("Example GraphRAG Result"):
                st.code("""
[
    {
        "transaction_id": "tx_456",
        "amount": 1199.99,
        "timestamp": "2025-03-15T14:22:33.000Z",
        "merchant_category": "electronics",
        "country": "USA",
        "device_type": "mobile",
        "fraud_pattern": "Multiple high-value purchases in short time",
        "pattern_type": "velocity_abuse"
    },
    {
        "transaction_id": "tx_789",
        "amount": 1499.99,
        "timestamp": "2025-02-28T09:45:12.000Z",
        "merchant_category": "electronics",
        "country": "Canada",
        "device_type": "desktop",
        "fraud_pattern": "Online transactions with suspicious patterns",
        "pattern_type": "card_not_present"
    },
    {
        "transaction_id": "tx_101",
        "amount": 1099.99,
        "timestamp": "2025-04-01T16:33:21.000Z",
        "merchant_category": "electronics",
        "country": "USA",
        "device_type": "mobile",
        "fraud_pattern": "Unusual location and spending pattern",
        "pattern_type": "identity_theft"
    }
]
                """, language="json")
    
    def render_dashboard(self):
        """Render the complete dashboard"""
        # Render header
        self.render_header()
        
        # Create tabs for different sections
        tabs = st.tabs([
            "Fraud Patterns", 
            "Transaction Investigation", 
            "Multi-Agent System", 
            "Agent Logs"
        ])
        
        with tabs[0]:
            # Render fraud patterns overview
            self.render_fraud_patterns_overview()
            
            # Render pattern details
            self.render_pattern_details()
        
        with tabs[1]:
            # Render transaction investigation
            self.render_transaction_investigation()
        
        with tabs[2]:
            # Create subtabs for different multi-agent visualizations
            subtabs = st.tabs(["MCP Visualization", "A2A Visualization", "ADK Visualization"])
            
            with subtabs[0]:
                # Render MCP visualization
                self.render_mcp_visualization()
            
            with subtabs[1]:
                # Render A2A visualization
                self.render_a2a_visualization()
            
            with subtabs[2]:
                # Render ADK visualization
                self.render_adk_visualization()
        
        with tabs[3]:
            # Render agent logs
            self.render_agent_logs()

def main():
    parser = argparse.ArgumentParser(description='Fraud Investigation Dashboard')
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
    dashboard = FraudInvestigationDashboard(
        neo4j_uri=args.neo4j_uri,
        neo4j_username=args.neo4j_username,
        neo4j_password=args.neo4j_password,
        data_dir=args.data_dir
    )
    
    # Render dashboard
    dashboard.render_dashboard()

if __name__ == "__main__":
    main()
