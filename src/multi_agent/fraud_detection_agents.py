#!/usr/bin/env python3
"""
Multi-Agent System with LangGraph for Credit Card Fraud Detection

This script implements a multi-agent system using LangGraph for orchestrating
the fraud detection workflow, with Model Context Protocol (MCP) for multi-modal
context handling and Agent-to-Agent (A2A) communication.
"""

import os
import json
import argparse
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools
from langgraph.checkpoint import MemorySaver

# Neo4j ADK
import sys
sys.path.append('/home/ubuntu/credit_card_fraud_detection/src/neo4j')
from neo4j_adk import Neo4jADK

# Model Context Protocol (MCP)
class ModelContextProtocol:
    """
    Implementation of Model Context Protocol (MCP) for handling multi-modal context
    in the fraud detection system.
    """
    
    def __init__(self):
        self.contexts = {}
    
    def create_context(self, context_id: str, context_type: str) -> Dict[str, Any]:
        """Create a new context with the specified type"""
        context = {
            "context_id": context_id,
            "context_type": context_type,
            "created_at": datetime.datetime.now().isoformat(),
            "content": {},
            "metadata": {},
            "references": []
        }
        self.contexts[context_id] = context
        return context
    
    def add_text_content(self, context_id: str, text: str, key: str = "text") -> None:
        """Add text content to a context"""
        if context_id not in self.contexts:
            raise ValueError(f"Context {context_id} does not exist")
        
        self.contexts[context_id]["content"][key] = text
    
    def add_image_content(self, context_id: str, image_path: str, key: str = "image") -> None:
        """Add image content to a context"""
        if context_id not in self.contexts:
            raise ValueError(f"Context {context_id} does not exist")
        
        self.contexts[context_id]["content"][key] = {"type": "image", "path": image_path}
    
    def add_structured_content(self, context_id: str, data: Dict[str, Any], key: str = "data") -> None:
        """Add structured data content to a context"""
        if context_id not in self.contexts:
            raise ValueError(f"Context {context_id} does not exist")
        
        self.contexts[context_id]["content"][key] = data
    
    def add_reference(self, context_id: str, ref_context_id: str) -> None:
        """Add a reference to another context"""
        if context_id not in self.contexts:
            raise ValueError(f"Context {context_id} does not exist")
        if ref_context_id not in self.contexts:
            raise ValueError(f"Referenced context {ref_context_id} does not exist")
        
        self.contexts[context_id]["references"].append(ref_context_id)
    
    def get_context(self, context_id: str) -> Dict[str, Any]:
        """Get a context by ID"""
        if context_id not in self.contexts:
            raise ValueError(f"Context {context_id} does not exist")
        
        return self.contexts[context_id]
    
    def get_all_contexts(self) -> Dict[str, Dict[str, Any]]:
        """Get all contexts"""
        return self.contexts
    
    def merge_contexts(self, context_ids: List[str], new_context_id: str, new_context_type: str) -> Dict[str, Any]:
        """Merge multiple contexts into a new context"""
        # Create new context
        new_context = self.create_context(new_context_id, new_context_type)
        
        # Merge content and references
        for context_id in context_ids:
            if context_id not in self.contexts:
                raise ValueError(f"Context {context_id} does not exist")
            
            context = self.contexts[context_id]
            
            # Merge content
            for key, value in context["content"].items():
                if key not in new_context["content"]:
                    new_context["content"][key] = value
                else:
                    # If key already exists, append with a numbered suffix
                    new_key = f"{key}_{context_id}"
                    new_context["content"][new_key] = value
            
            # Add references
            for ref in context["references"]:
                if ref not in new_context["references"]:
                    new_context["references"].append(ref)
            
            # Add original context as reference
            if context_id not in new_context["references"]:
                new_context["references"].append(context_id)
        
        return new_context

# Agent-to-Agent (A2A) Communication
class A2ACommunication:
    """
    Implementation of Agent-to-Agent (A2A) communication for the fraud detection system.
    """
    
    def __init__(self):
        self.messages = []
        self.subscriptions = {}
    
    def send_message(self, sender: str, receiver: str, message_type: str, content: Dict[str, Any], 
                     priority: str = "medium", context_references: List[str] = None) -> Dict[str, Any]:
        """Send a message from one agent to another"""
        message_id = f"msg_{len(self.messages) + 1:06d}"
        timestamp = datetime.datetime.now().isoformat()
        
        message = {
            "message_id": message_id,
            "sender": sender,
            "receiver": receiver,
            "message_type": message_type,
            "priority": priority,
            "content": content,
            "timestamp": timestamp,
            "context_references": context_references or [],
            "status": "sent"
        }
        
        self.messages.append(message)
        
        # Notify subscribers
        if receiver in self.subscriptions:
            for callback in self.subscriptions[receiver]:
                callback(message)
        
        return message
    
    def broadcast_message(self, sender: str, message_type: str, content: Dict[str, Any],
                         priority: str = "medium", context_references: List[str] = None) -> List[Dict[str, Any]]:
        """Broadcast a message to all subscribers"""
        timestamp = datetime.datetime.now().isoformat()
        sent_messages = []
        
        for receiver in self.subscriptions.keys():
            if receiver != sender:  # Don't send to self
                message_id = f"msg_{len(self.messages) + 1:06d}"
                
                message = {
                    "message_id": message_id,
                    "sender": sender,
                    "receiver": receiver,
                    "message_type": message_type,
                    "priority": priority,
                    "content": content,
                    "timestamp": timestamp,
                    "context_references": context_references or [],
                    "status": "sent"
                }
                
                self.messages.append(message)
                sent_messages.append(message)
                
                # Notify subscriber
                for callback in self.subscriptions[receiver]:
                    callback(message)
        
        return sent_messages
    
    def subscribe(self, agent_id: str, callback) -> None:
        """Subscribe to messages for a specific agent"""
        if agent_id not in self.subscriptions:
            self.subscriptions[agent_id] = []
        
        self.subscriptions[agent_id].append(callback)
    
    def get_messages(self, agent_id: str = None, message_type: str = None, 
                    since: str = None, limit: int = None) -> List[Dict[str, Any]]:
        """Get messages filtered by agent ID, message type, and timestamp"""
        filtered_messages = self.messages
        
        if agent_id:
            filtered_messages = [m for m in filtered_messages if m["sender"] == agent_id or m["receiver"] == agent_id]
        
        if message_type:
            filtered_messages = [m for m in filtered_messages if m["message_type"] == message_type]
        
        if since:
            filtered_messages = [m for m in filtered_messages if m["timestamp"] > since]
        
        # Sort by timestamp (newest first)
        filtered_messages = sorted(filtered_messages, key=lambda m: m["timestamp"], reverse=True)
        
        if limit:
            filtered_messages = filtered_messages[:limit]
        
        return filtered_messages
    
    def mark_as_read(self, message_id: str) -> None:
        """Mark a message as read"""
        for message in self.messages:
            if message["message_id"] == message_id:
                message["status"] = "read"
                break

# Multi-Agent System
class FraudDetectionAgentSystem:
    """
    Multi-agent system for credit card fraud detection using LangGraph for orchestration,
    MCP for context handling, and A2A for communication.
    """
    
    def __init__(self, neo4j_uri: str, neo4j_username: str, neo4j_password: str):
        # Initialize components
        self.mcp = ModelContextProtocol()
        self.a2a = A2ACommunication()
        self.neo4j_adk = Neo4jADK(neo4j_uri, neo4j_username, neo4j_password)
        
        # Initialize agent states
        self.agent_states = {}
        
        # Create the agent workflow graph
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for the multi-agent system"""
        # Define the workflow graph
        workflow = StateGraph(StateGraph.construct_init_state)
        
        # Add nodes for each agent
        workflow.add_node("transaction_analysis_agent", self.transaction_analysis_agent)
        workflow.add_node("pattern_detection_agent", self.pattern_detection_agent)
        workflow.add_node("investigation_agent", self.investigation_agent)
        workflow.add_node("decision_agent", self.decision_agent)
        workflow.add_node("user_profile_agent", self.user_profile_agent)
        workflow.add_node("merchant_risk_agent", self.merchant_risk_agent)
        workflow.add_node("feedback_collection_agent", self.feedback_collection_agent)
        workflow.add_node("learning_agent", self.learning_agent)
        
        # Define the edges (workflow transitions)
        workflow.add_edge("transaction_analysis_agent", "pattern_detection_agent")
        workflow.add_conditional_edges(
            "pattern_detection_agent",
            self._should_investigate,
            {
                True: "investigation_agent",
                False: "decision_agent"
            }
        )
        workflow.add_edge("investigation_agent", "decision_agent")
        workflow.add_edge("decision_agent", "feedback_collection_agent")
        workflow.add_edge("feedback_collection_agent", "learning_agent")
        workflow.add_edge("learning_agent", END)
        
        # Add conditional callbacks to user_profile_agent and merchant_risk_agent
        workflow.add_conditional_edges(
            "transaction_analysis_agent",
            self._needs_user_profile,
            {
                True: "user_profile_agent",
                False: "pattern_detection_agent"
            }
        )
        workflow.add_conditional_edges(
            "user_profile_agent",
            lambda x: True,  # Always proceed to pattern detection after user profile
            {
                True: "pattern_detection_agent"
            }
        )
        workflow.add_conditional_edges(
            "transaction_analysis_agent",
            self._needs_merchant_risk,
            {
                True: "merchant_risk_agent",
                False: "pattern_detection_agent"
            }
        )
        workflow.add_conditional_edges(
            "merchant_risk_agent",
            lambda x: True,  # Always proceed to pattern detection after merchant risk
            {
                True: "pattern_detection_agent"
            }
        )
        
        # Compile the workflow
        workflow.compile()
        
        return workflow
    
    # Conditional functions for workflow routing
    def _should_investigate(self, state: Dict[str, Any]) -> bool:
        """Determine if a transaction needs further investigation"""
        # Check if the fraud score is in the medium range (requires investigation)
        fraud_score = state.get("fraud_score", 0)
        return 0.3 <= fraud_score < 0.7
    
    def _needs_user_profile(self, state: Dict[str, Any]) -> bool:
        """Determine if user profile information is needed"""
        # Check if the transaction is unusual for the user
        amount = state.get("transaction", {}).get("amount", 0)
        user_id = state.get("transaction", {}).get("user_id")
        
        if not user_id:
            return False
        
        # Get user profile
        try:
            user_profile = self.neo4j_adk.get_user_profile(user_id)
            typical_amount = float(user_profile.get("typical_amount", 0))
            
            # If amount is significantly different from typical amount, get user profile
            return amount > typical_amount * 2 or amount < typical_amount * 0.2
        except:
            return True  # If error, get user profile to be safe
    
    def _needs_merchant_risk(self, state: Dict[str, Any]) -> bool:
        """Determine if merchant risk information is needed"""
        # Check if the merchant is high risk or new to the user
        merchant_id = state.get("transaction", {}).get("merchant_id")
        user_id = state.get("transaction", {}).get("user_id")
        
        if not merchant_id or not user_id:
            return False
        
        # Get user profile to check if merchant is frequently visited
        try:
            user_profile = self.neo4j_adk.get_user_profile(user_id)
            frequent_merchants = user_profile.get("frequent_merchants", [])
            
            # If merchant is not in frequent merchants, check merchant risk
            for merchant in frequent_merchants:
                if merchant.get("merchant", {}).get("merchantId") == merchant_id:
                    return False
            
            return True  # Not a frequent merchant, check risk
        except:
            return True  # If error, check merchant risk to be safe
    
    # Agent implementations
    def transaction_analysis_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transaction Analysis Agent: Analyzes individual transactions for fraud indicators
        """
        print("Transaction Analysis Agent: Processing transaction...")
        
        # Get transaction data
        transaction = state.get("transaction", {})
        transaction_id = transaction.get("transaction_id")
        
        if not transaction_id:
            return {**state, "error": "No transaction ID provided"}
        
        # Create context for this transaction
        context_id = f"ctx_tx_{transaction_id}"
        self.mcp.create_context(context_id, "transaction_analysis")
        self.mcp.add_structured_content(context_id, transaction)
        
        # Query Neo4j for transaction details
        try:
            tx_details = self.neo4j_adk.get_transaction_by_id(transaction_id)
            if tx_details:
                self.mcp.add_structured_content(context_id, tx_details, "transaction_details")
        except Exception as e:
            print(f"Error querying Neo4j: {e}")
        
        # Analyze transaction for basic fraud indicators
        amount = float(transaction.get("amount", 0))
        is_online = transaction.get("is_online", False)
        payment_method = transaction.get("payment_method", "")
        
        # Calculate initial fraud score based on simple rules
        fraud_score = 0.0
        fraud_indicators = []
        
        # Check amount
        if amount > 1000:
            fraud_score += 0.2
            fraud_indicators.append("high_amount")
        
        # Check payment method
        if is_online and payment_method == "manual_entry":
            fraud_score += 0.3
            fraud_indicators.append("online_manual_entry")
        
        # Check device risk if online
        if is_online and "device_context" in transaction:
            device_risk = float(transaction.get("device_context", {}).get("risk_score", 0))
            if device_risk > 0.7:
                fraud_score += 0.3
                fraud_indicators.append("high_risk_device")
            
            # Check IP risk
            ip_is_proxy = transaction.get("device_context", {}).get("ip_is_proxy", False)
            ip_is_vpn = transaction.get("device_context", {}).get("ip_is_vpn", False)
            ip_is_tor = transaction.get("device_context", {}).get("ip_is_tor", False)
            
            if ip_is_proxy or ip_is_vpn or ip_is_tor:
                fraud_score += 0.3
                fraud_indicators.append("suspicious_ip")
        
        # Add analysis results to context
        analysis_results = {
            "initial_fraud_score": fraud_score,
            "fraud_indicators": fraud_indicators
        }
        self.mcp.add_structured_content(context_id, analysis_results, "analysis_results")
        
        # Send message to other agents
        self.a2a.send_message(
            sender="transaction_analysis_agent",
            receiver="pattern_detection_agent",
            message_type="transaction_analyzed",
            content={
                "transaction_id": transaction_id,
                "fraud_score": fraud_score,
                "fraud_indicators": fraud_indicators
            },
            context_references=[context_id]
        )
        
        # Update state
        return {
            **state,
            "transaction_id": transaction_id,
            "fraud_score": fraud_score,
            "fraud_indicators": fraud_indicators,
            "context_id": context_id
        }
    
    def pattern_detection_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pattern Detection Agent: Identifies patterns across multiple transactions
        """
        print("Pattern Detection Agent: Detecting patterns...")
        
        # Get transaction data
        transaction_id = state.get("transaction_id")
        initial_fraud_score = state.get("fraud_score", 0)
        
        if not transaction_id:
            return {**state, "error": "No transaction ID provided"}
        
        # Create context for pattern detection
        context_id = f"ctx_pattern_{transaction_id}"
        self.mcp.create_context(context_id, "pattern_detection")
        
        # Reference the transaction analysis context
        tx_context_id = state.get("context_id")
        if tx_context_id:
            self.mcp.add_reference(context_id, tx_context_id)
        
        # Query Neo4j for pattern detection
        patterns_detected = []
        pattern_score = 0.0
        
        try:
            # Check for velocity anomalies
            velocity_anomalies = self.neo4j_adk.detect_velocity_anomalies()
            for anomaly in velocity_anomalies:
                if transaction_id in anomaly.get("subsequent_transaction_ids", []) or transaction_id == anomaly.get("first_transaction_id"):
                    patterns_detected.append("velocity_anomaly")
                    pattern_score += 0.3
                    break
            
            # Check for location anomalies
            location_anomalies = self.neo4j_adk.detect_location_anomalies()
            for anomaly in location_anomalies:
                if transaction_id == anomaly.get("transaction1_id") or transaction_id == anomaly.get("transaction2_id"):
                    patterns_detected.append("location_anomaly")
                    pattern_score += 0.4
                    break
            
            # Check for unusual merchant activity
            unusual_merchants = self.neo4j_adk.detect_unusual_merchant_activity()
            for activity in unusual_merchants:
                if transaction_id == activity.get("transaction_id"):
                    patterns_detected.append("unusual_merchant")
                    pattern_score += 0.2
                    break
            
            # Check for high risk devices
            high_risk_devices = self.neo4j_adk.detect_high_risk_devices()
            for device in high_risk_devices:
                if transaction_id == device.get("transaction_id"):
                    patterns_detected.append("high_risk_device")
                    pattern_score += 0.3
                    break
            
            # Get similar fraud patterns
            similar_patterns = self.neo4j_adk.get_similar_fraud_patterns(transaction_id)
            if similar_patterns:
                for pattern in similar_patterns:
                    patterns_detected.append(f"similar_to_{pattern.get('pattern_type')}")
                    pattern_score += float(pattern.get("similarity_score", 0)) * 0.5
        
        except Exception as e:
            print(f"Error in pattern detection: {e}")
        
        # Add pattern detection results to context
        pattern_results = {
            "patterns_detected": patterns_detected,
            "pattern_score": pattern_score
        }
        self.mcp.add_structured_content(context_id, pattern_results, "pattern_results")
        
        # Calculate combined fraud score
        combined_score = min(0.95, initial_fraud_score + pattern_score)
        
        # Send message to other agents
        self.a2a.send_message(
            sender="pattern_detection_agent",
            receiver="investigation_agent" if self._should_investigate({**state, "fraud_score": combined_score}) else "decision_agent",
            message_type="patterns_detected",
            content={
                "transaction_id": transaction_id,
                "patterns_detected": patterns_detected,
                "pattern_score": pattern_score,
                "combined_score": combined_score
            },
            context_references=[context_id, tx_context_id] if tx_context_id else [context_id]
        )
        
        # Update state
        return {
            **state,
            "patterns_detected": patterns_detected,
            "pattern_score": pattern_score,
            "fraud_score": combined_score,
            "pattern_context_id": context_id
        }
    
    def investigation_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Investigation Agent: Conducts deeper investigation of suspicious activities
        """
        print("Investigation Agent: Investigating suspicious transaction...")
        
        # Get transaction data
        transaction_id = state.get("transaction_id")
        combined_score = state.get("fraud_score", 0)
        
        if not transaction_id:
            return {**state, "error": "No transaction ID provided"}
        
        # Create context for investigation
        context_id = f"ctx_investigation_{transaction_id}"
        self.mcp.create_context(context_id, "investigation")
        
        # Reference previous contexts
        tx_context_id = state.get("context_id")
        pattern_context_id = state.get("pattern_context_id")
        
        if tx_context_id:
            self.mcp.add_reference(context_id, tx_context_id)
        if pattern_context_id:
            self.mcp.add_reference(context_id, pattern_context_id)
        
        # Perform deep investigation
        investigation_results = {}
        investigation_score_adjustment = 0.0
        
        try:
            # Get transaction chain
            transaction_chain = self.neo4j_adk.get_transaction_chain(transaction_id)
            investigation_results["transaction_chain"] = transaction_chain
            
            # Check for fraud patterns in chain
            fraud_in_chain = any(tx.get("is_fraudulent") for tx in transaction_chain)
            if fraud_in_chain:
                investigation_score_adjustment += 0.2
            
            # Get GraphRAG knowledge retrieval
            tx_details = self.neo4j_adk.get_transaction_by_id(transaction_id)
            if tx_details:
                transaction = tx_details.get("transaction", {})
                merchant = tx_details.get("merchant", {})
                
                # Prepare properties for GraphRAG
                properties = {
                    "amount": transaction.get("amount"),
                    "is_online": transaction.get("isOnline"),
                    "merchant_category": merchant.get("category")
                }
                
                # Retrieve relevant knowledge
                knowledge = self.neo4j_adk.retrieve_fraud_knowledge(properties)
                investigation_results["similar_fraud_cases"] = knowledge
                
                # Adjust score based on similar cases
                if knowledge and len(knowledge) > 2:  # If multiple similar fraud cases exist
                    investigation_score_adjustment += 0.15
            
            # Get user fraud risk
            user_id = state.get("transaction", {}).get("user_id")
            if user_id:
                user_risk = self.neo4j_adk.get_user_fraud_risk(user_id)
                investigation_results["user_risk"] = user_risk
                
                if user_risk and user_risk.get("calculated_risk_score", 0) > 0.7:
                    investigation_score_adjustment += 0.1
        
        except Exception as e:
            print(f"Error in investigation: {e}")
        
        # Add investigation results to context
        self.mcp.add_structured_content(context_id, investigation_results, "investigation_results")
        
        # Calculate final fraud score
        final_score = min(0.95, combined_score + investigation_score_adjustment)
        
        # Send message to decision agent
        self.a2a.send_message(
            sender="investigation_agent",
            receiver="decision_agent",
            message_type="investigation_completed",
            content={
                "transaction_id": transaction_id,
                "investigation_results": investigation_results,
                "score_adjustment": investigation_score_adjustment,
                "final_score": final_score
            },
            context_references=[context_id, tx_context_id, pattern_context_id] if tx_context_id and pattern_context_id else [context_id]
        )
        
        # Update state
        return {
            **state,
            "investigation_results": investigation_results,
            "investigation_score_adjustment": investigation_score_adjustment,
            "fraud_score": final_score,
            "investigation_context_id": context_id
        }
    
    def decision_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decision Agent: Makes final fraud determination
        """
        print("Decision Agent: Making final determination...")
        
        # Get transaction data
        transaction_id = state.get("transaction_id")
        final_score = state.get("fraud_score", 0)
        
        if not transaction_id:
            return {**state, "error": "No transaction ID provided"}
        
        # Create context for decision
        context_id = f"ctx_decision_{transaction_id}"
        self.mcp.create_context(context_id, "decision")
        
        # Reference previous contexts
        tx_context_id = state.get("context_id")
        pattern_context_id = state.get("pattern_context_id")
        investigation_context_id = state.get("investigation_context_id")
        
        if tx_context_id:
            self.mcp.add_reference(context_id, tx_context_id)
        if pattern_context_id:
            self.mcp.add_reference(context_id, pattern_context_id)
        if investigation_context_id:
            self.mcp.add_reference(context_id, investigation_context_id)
        
        # Make decision based on final score
        decision = "approve"
        reason = "Transaction appears legitimate"
        fraud_pattern = None
        
        if final_score >= 0.8:
            decision = "deny"
            reason = "High probability of fraud"
            # Determine most likely fraud pattern
            patterns = state.get("patterns_detected", [])
            if patterns:
                if "velocity_anomaly" in patterns:
                    fraud_pattern = "velocity_abuse"
                elif "location_anomaly" in patterns:
                    fraud_pattern = "location_anomaly"
                elif "unusual_merchant" in patterns:
                    fraud_pattern = "merchant_fraud"
                elif "high_risk_device" in patterns:
                    fraud_pattern = "card_not_present"
                else:
                    for pattern in patterns:
                        if pattern.startswith("similar_to_"):
                            fraud_pattern = pattern.replace("similar_to_", "")
                            break
        elif final_score >= 0.5:
            decision = "flag_for_review"
            reason = "Suspicious activity detected"
        elif final_score >= 0.3:
            decision = "additional_authentication"
            reason = "Unusual activity requires verification"
        
        # Update transaction in Neo4j
        try:
            if decision == "deny":
                self.neo4j_adk.update_transaction_fraud_status(
                    transaction_id=transaction_id,
                    is_fraudulent=True,
                    fraud_score=final_score,
                    fraud_pattern=fraud_pattern
                )
                
                # Create fraud alert
                self.neo4j_adk.create_fraud_alert(
                    transaction_id=transaction_id,
                    alert_type="fraud_detected",
                    severity="high",
                    description=f"Fraud detected: {reason}. Pattern: {fraud_pattern}"
                )
            elif decision == "flag_for_review":
                # Create alert for review
                self.neo4j_adk.create_fraud_alert(
                    transaction_id=transaction_id,
                    alert_type="suspicious_activity",
                    severity="medium",
                    description=f"Suspicious activity: {reason}"
                )
        except Exception as e:
            print(f"Error updating Neo4j: {e}")
        
        # Add decision to context
        decision_result = {
            "decision": decision,
            "reason": reason,
            "fraud_score": final_score,
            "fraud_pattern": fraud_pattern
        }
        self.mcp.add_structured_content(context_id, decision_result, "decision_result")
        
        # Send message to feedback collection agent
        self.a2a.send_message(
            sender="decision_agent",
            receiver="feedback_collection_agent",
            message_type="decision_made",
            content=decision_result,
            context_references=[context_id]
        )
        
        # Update state
        return {
            **state,
            "decision": decision,
            "reason": reason,
            "fraud_pattern": fraud_pattern,
            "decision_context_id": context_id
        }
    
    def user_profile_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        User Profile Agent: Maintains and analyzes user behavior profiles
        """
        print("User Profile Agent: Analyzing user profile...")
        
        # Get transaction data
        transaction = state.get("transaction", {})
        user_id = transaction.get("user_id")
        transaction_id = transaction.get("transaction_id")
        
        if not user_id or not transaction_id:
            return state  # No changes if missing data
        
        # Create context for user profile
        context_id = f"ctx_user_{transaction_id}"
        self.mcp.create_context(context_id, "user_profile")
        
        # Reference transaction context
        tx_context_id = state.get("context_id")
        if tx_context_id:
            self.mcp.add_reference(context_id, tx_context_id)
        
        # Get user profile from Neo4j
        user_profile = None
        try:
            user_profile = self.neo4j_adk.get_user_profile(user_id)
        except Exception as e:
            print(f"Error getting user profile: {e}")
        
        if not user_profile:
            return state  # No changes if profile not found
        
        # Add user profile to context
        self.mcp.add_structured_content(context_id, user_profile, "user_profile")
        
        # Analyze transaction against user profile
        amount = float(transaction.get("amount", 0))
        typical_amount = float(user_profile.get("typical_amount", 0))
        
        profile_analysis = {
            "amount_ratio": amount / typical_amount if typical_amount > 0 else float('inf'),
            "is_frequent_merchant": False,
            "is_typical_location": False
        }
        
        # Check if merchant is frequently visited
        merchant_id = transaction.get("merchant_id")
        for merchant in user_profile.get("frequent_merchants", []):
            if merchant.get("merchant", {}).get("merchantId") == merchant_id:
                profile_analysis["is_frequent_merchant"] = True
                break
        
        # Check if location is typical
        location_id = transaction.get("location_id")
        for location in user_profile.get("typical_locations", []):
            if location.get("location", {}).get("locationId") == location_id:
                profile_analysis["is_typical_location"] = True
                break
        
        # Add profile analysis to context
        self.mcp.add_structured_content(context_id, profile_analysis, "profile_analysis")
        
        # Calculate risk adjustment based on profile
        risk_adjustment = 0.0
        
        if profile_analysis["amount_ratio"] > 5:
            risk_adjustment += 0.3
        elif profile_analysis["amount_ratio"] > 2:
            risk_adjustment += 0.1
        
        if not profile_analysis["is_frequent_merchant"]:
            risk_adjustment += 0.1
        
        if not profile_analysis["is_typical_location"]:
            risk_adjustment += 0.2
        
        # Send message to transaction analysis agent
        self.a2a.send_message(
            sender="user_profile_agent",
            receiver="transaction_analysis_agent",
            message_type="profile_analyzed",
            content={
                "user_id": user_id,
                "transaction_id": transaction_id,
                "profile_analysis": profile_analysis,
                "risk_adjustment": risk_adjustment
            },
            context_references=[context_id]
        )
        
        # Update state
        return {
            **state,
            "user_profile": user_profile,
            "profile_analysis": profile_analysis,
            "profile_risk_adjustment": risk_adjustment,
            "fraud_score": state.get("fraud_score", 0) + risk_adjustment,
            "user_profile_context_id": context_id
        }
    
    def merchant_risk_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merchant Risk Agent: Assesses merchant risk levels
        """
        print("Merchant Risk Agent: Assessing merchant risk...")
        
        # Get transaction data
        transaction = state.get("transaction", {})
        merchant_id = transaction.get("merchant_id")
        transaction_id = transaction.get("transaction_id")
        
        if not merchant_id or not transaction_id:
            return state  # No changes if missing data
        
        # Create context for merchant risk
        context_id = f"ctx_merchant_{transaction_id}"
        self.mcp.create_context(context_id, "merchant_risk")
        
        # Reference transaction context
        tx_context_id = state.get("context_id")
        if tx_context_id:
            self.mcp.add_reference(context_id, tx_context_id)
        
        # Get merchant details from Neo4j
        merchant_details = None
        try:
            tx_details = self.neo4j_adk.get_transaction_by_id(transaction_id)
            if tx_details:
                merchant_details = tx_details.get("merchant", {})
        except Exception as e:
            print(f"Error getting merchant details: {e}")
        
        if not merchant_details:
            return state  # No changes if details not found
        
        # Add merchant details to context
        self.mcp.add_structured_content(context_id, merchant_details, "merchant_details")
        
        # Analyze merchant risk
        merchant_risk = float(merchant_details.get("riskScore", 0))
        is_high_risk = merchant_details.get("isHighRisk", False)
        fraud_rate = float(merchant_details.get("fraudRate", 0))
        
        risk_analysis = {
            "merchant_risk_score": merchant_risk,
            "is_high_risk": is_high_risk,
            "fraud_rate": fraud_rate
        }
        
        # Add risk analysis to context
        self.mcp.add_structured_content(context_id, risk_analysis, "risk_analysis")
        
        # Calculate risk adjustment based on merchant
        risk_adjustment = 0.0
        
        if is_high_risk:
            risk_adjustment += 0.3
        elif merchant_risk > 0.7:
            risk_adjustment += 0.2
        elif merchant_risk > 0.4:
            risk_adjustment += 0.1
        
        if fraud_rate > 0.05:
            risk_adjustment += 0.3
        elif fraud_rate > 0.01:
            risk_adjustment += 0.1
        
        # Send message to transaction analysis agent
        self.a2a.send_message(
            sender="merchant_risk_agent",
            receiver="transaction_analysis_agent",
            message_type="merchant_analyzed",
            content={
                "merchant_id": merchant_id,
                "transaction_id": transaction_id,
                "risk_analysis": risk_analysis,
                "risk_adjustment": risk_adjustment
            },
            context_references=[context_id]
        )
        
        # Update state
        return {
            **state,
            "merchant_details": merchant_details,
            "merchant_risk_analysis": risk_analysis,
            "merchant_risk_adjustment": risk_adjustment,
            "fraud_score": state.get("fraud_score", 0) + risk_adjustment,
            "merchant_risk_context_id": context_id
        }
    
    def feedback_collection_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Feedback Collection Agent: Gathers feedback on detection accuracy
        """
        print("Feedback Collection Agent: Collecting feedback...")
        
        # Get decision data
        transaction_id = state.get("transaction_id")
        decision = state.get("decision")
        
        if not transaction_id or not decision:
            return state  # No changes if missing data
        
        # Create context for feedback
        context_id = f"ctx_feedback_{transaction_id}"
        self.mcp.create_context(context_id, "feedback_collection")
        
        # Reference decision context
        decision_context_id = state.get("decision_context_id")
        if decision_context_id:
            self.mcp.add_reference(context_id, decision_context_id)
        
        # In a real system, this would collect feedback from users or analysts
        # For this demo, we'll simulate feedback
        
        # Simulate feedback based on decision
        feedback = {
            "transaction_id": transaction_id,
            "decision": decision,
            "feedback_source": "simulated",
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        if decision == "deny":
            # Simulate feedback for denied transactions
            feedback["is_correct"] = random.random() < 0.9  # 90% correct
            feedback["feedback_type"] = "true_positive" if feedback["is_correct"] else "false_positive"
            feedback["confidence"] = random.uniform(0.7, 1.0)
        elif decision == "approve":
            # Simulate feedback for approved transactions
            feedback["is_correct"] = random.random() < 0.95  # 95% correct
            feedback["feedback_type"] = "true_negative" if feedback["is_correct"] else "false_negative"
            feedback["confidence"] = random.uniform(0.8, 1.0)
        else:
            # Simulate feedback for flagged or additional auth
            feedback["is_correct"] = random.random() < 0.7  # 70% correct
            feedback["feedback_type"] = "true_positive" if feedback["is_correct"] else "false_positive"
            feedback["confidence"] = random.uniform(0.5, 0.9)
        
        # Add feedback to context
        self.mcp.add_structured_content(context_id, feedback, "feedback")
        
        # Send message to learning agent
        self.a2a.send_message(
            sender="feedback_collection_agent",
            receiver="learning_agent",
            message_type="feedback_collected",
            content=feedback,
            context_references=[context_id]
        )
        
        # Update state
        return {
            **state,
            "feedback": feedback,
            "feedback_context_id": context_id
        }
    
    def learning_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Learning Agent: Updates models based on feedback and new patterns
        """
        print("Learning Agent: Processing feedback for model updates...")
        
        # Get feedback data
        feedback = state.get("feedback", {})
        transaction_id = state.get("transaction_id")
        
        if not feedback or not transaction_id:
            return state  # No changes if missing data
        
        # Create context for learning
        context_id = f"ctx_learning_{transaction_id}"
        self.mcp.create_context(context_id, "learning")
        
        # Reference feedback context
        feedback_context_id = state.get("feedback_context_id")
        if feedback_context_id:
            self.mcp.add_reference(context_id, feedback_context_id)
        
        # In a real system, this would update models based on feedback
        # For this demo, we'll simulate learning
        
        # Simulate learning process
        learning_result = {
            "transaction_id": transaction_id,
            "feedback_processed": True,
            "model_updated": random.random() < 0.8,  # 80% chance of model update
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add learning result to context
        self.mcp.add_structured_content(context_id, learning_result, "learning_result")
        
        # Broadcast message to all agents
        self.a2a.broadcast_message(
            sender="learning_agent",
            message_type="model_updated",
            content={
                "transaction_id": transaction_id,
                "model_updated": learning_result["model_updated"],
                "update_type": feedback.get("feedback_type")
            },
            context_references=[context_id]
        )
        
        # Update state
        return {
            **state,
            "learning_result": learning_result,
            "learning_context_id": context_id
        }
    
    def process_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Process a transaction through the multi-agent system"""
        print(f"Processing transaction {transaction.get('transaction_id')}...")
        
        # Initialize state with transaction
        initial_state = {"transaction": transaction}
        
        # Run the workflow
        try:
            result = self.workflow.invoke(initial_state)
            return result
        except Exception as e:
            print(f"Error processing transaction: {e}")
            return {"error": str(e), "transaction": transaction}

def main():
    parser = argparse.ArgumentParser(description='Multi-Agent System for Credit Card Fraud Detection')
    parser.add_argument('--neo4j-uri', type=str, default='bolt://localhost:7687',
                        help='Neo4j URI')
    parser.add_argument('--neo4j-username', type=str, default='neo4j',
                        help='Neo4j username')
    parser.add_argument('--neo4j-password', type=str, default='password',
                        help='Neo4j password')
    parser.add_argument('--transaction-id', type=str, required=True,
                        help='Transaction ID to process')
    
    args = parser.parse_args()
    
    # Initialize the multi-agent system
    agent_system = FraudDetectionAgentSystem(
        neo4j_uri=args.neo4j_uri,
        neo4j_username=args.neo4j_username,
        neo4j_password=args.neo4j_password
    )
    
    # Get transaction from Neo4j
    transaction = None
    try:
        tx_details = agent_system.neo4j_adk.get_transaction_by_id(args.transaction_id)
        if tx_details:
            transaction = tx_details.get("transaction", {})
    except Exception as e:
        print(f"Error getting transaction: {e}")
    
    if not transaction:
        print(f"Transaction {args.transaction_id} not found")
        return
    
    # Process the transaction
    result = agent_system.process_transaction(transaction)
    
    # Print result
    print("\nProcessing Result:")
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()
