#!/usr/bin/env python3
"""
Credit Card Fraud Detection - Test Data Generator

This script generates synthetic credit card transaction data for testing the fraud detection system.
It creates normal transaction patterns as well as various fraud patterns to demonstrate the system's capabilities.
"""

import pandas as pd
import numpy as np
import random
import uuid
import datetime
import json
import os
from faker import Faker

# Initialize Faker for generating realistic data
fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# Constants
NUM_USERS = 1000
NUM_CARDS_PER_USER_MAX = 3
NUM_MERCHANTS = 500
NUM_LOCATIONS = 200
NUM_DEVICES = 1500
NUM_NORMAL_TRANSACTIONS = 50000
NUM_FRAUDULENT_TRANSACTIONS = 5000
OUTPUT_DIR = "/home/ubuntu/credit_card_fraud_detection/data"

# Merchant categories
MERCHANT_CATEGORIES = [
    "Grocery", "Restaurant", "Gas Station", "Online Retail", "Department Store",
    "Electronics", "Travel", "Hotel", "Entertainment", "Healthcare",
    "Utilities", "Education", "Automotive", "Home Improvement", "Clothing",
    "Jewelry", "Sporting Goods", "Books", "Music", "Furniture", 
    "Money Transfer", "Financial Services", "Digital Goods"
]

# Payment methods
PAYMENT_METHODS = ["chip", "swipe", "online", "contactless", "manual_entry"]

# Transaction types
TRANSACTION_TYPES = ["purchase", "refund", "authorization", "adjustment", "cash_advance"]

# Card types
CARD_TYPES = ["Visa", "Mastercard", "Amex", "Discover"]

# Countries (with bias towards US)
COUNTRIES = ["US", "US", "US", "US", "CA", "CA", "UK", "UK", "FR", "DE", "JP", "AU", "BR", "IN", "MX"]

# User segments
USER_SEGMENTS = ["standard", "premium", "business", "student"]

# Device types
DEVICE_TYPES = ["mobile_android", "mobile_ios", "desktop_windows", "desktop_mac", "desktop_linux", "tablet"]

# Fraud patterns
FRAUD_PATTERNS = [
    "card_testing",           # Multiple small transactions in short time
    "identity_theft",         # Unusual location and spending pattern
    "account_takeover",       # Sudden change in behavior
    "card_not_present",       # Online transactions with suspicious patterns
    "merchant_fraud",         # Transactions at high-risk merchants
    "application_fraud",      # New account with unusual activity
    "transaction_laundering", # Complex chain of transactions
    "velocity_abuse",         # Many transactions in short time
    "location_anomaly",       # Transactions in multiple distant locations
    "amount_anomaly"          # Unusually large transaction amounts
]

class DataGenerator:
    def __init__(self):
        self.users = []
        self.cards = []
        self.merchants = []
        self.locations = []
        self.devices = []
        self.ip_addresses = []
        self.normal_transactions = []
        self.fraudulent_transactions = []
        
    def generate_users(self):
        """Generate synthetic user data"""
        print("Generating user data...")
        for i in range(NUM_USERS):
            user_id = f"user_{i:06d}"
            user = {
                "user_id": user_id,
                "name": fake.name(),
                "email": fake.email(),
                "phone": fake.phone_number(),
                "risk_score": round(np.random.beta(1, 10) * 100) / 100,  # Mostly low risk
                "account_creation_date": fake.date_time_between(start_date="-5y", end_date="now").isoformat(),
                "last_activity_date": fake.date_time_between(start_date="-1m", end_date="now").isoformat(),
                "fraud_history": random.random() < 0.02,  # 2% have fraud history
                "kyc_verified": random.random() < 0.95,   # 95% are verified
                "segment": random.choice(USER_SEGMENTS),
                "typical_transaction_amount": round(np.random.lognormal(4, 1), 2),  # Typical spending amount
                "typical_merchants": random.sample(range(NUM_MERCHANTS), min(10, random.randint(3, 15))),
                "typical_locations": random.sample(range(NUM_LOCATIONS), min(5, random.randint(1, 8)))
            }
            self.users.append(user)
        
        # Save to file
        df_users = pd.DataFrame(self.users)
        df_users.to_csv(f"{OUTPUT_DIR}/users.csv", index=False)
        print(f"Generated {len(self.users)} users")
        
    def generate_cards(self):
        """Generate synthetic credit card data"""
        print("Generating card data...")
        for user_idx, user in enumerate(self.users):
            # Each user has 1-3 cards
            num_cards = random.randint(1, NUM_CARDS_PER_USER_MAX)
            for j in range(num_cards):
                issue_date = datetime.datetime.fromisoformat(user["account_creation_date"])
                if j > 0:
                    # Additional cards issued later
                    issue_date = fake.date_time_between(start_date=issue_date, end_date="now")
                
                expiry_date = issue_date + datetime.timedelta(days=365*3 + random.randint(0, 365))
                
                card_id = f"card_{len(self.cards):06d}"
                card = {
                    "card_id": card_id,
                    "user_id": user["user_id"],
                    "card_type": random.choice(CARD_TYPES),
                    "issue_date": issue_date.isoformat(),
                    "expiry_date": expiry_date.isoformat(),
                    "last_four_digits": f"{random.randint(1000, 9999)}",
                    "is_active": random.random() < 0.95,  # 95% are active
                    "is_blocked": random.random() < 0.03,  # 3% are blocked
                    "credit_limit": random.choice([1000, 2000, 3000, 5000, 10000, 15000, 20000, 50000]),
                    "available_credit": None  # Will be calculated later
                }
                
                # Calculate available credit
                if card["is_active"] and not card["is_blocked"]:
                    utilization = random.uniform(0, 0.8)  # 0-80% utilization
                    card["available_credit"] = round(card["credit_limit"] * (1 - utilization), 2)
                else:
                    card["available_credit"] = 0
                
                self.cards.append(card)
        
        # Save to file
        df_cards = pd.DataFrame(self.cards)
        df_cards.to_csv(f"{OUTPUT_DIR}/cards.csv", index=False)
        print(f"Generated {len(self.cards)} cards")
        
    def generate_merchants(self):
        """Generate synthetic merchant data"""
        print("Generating merchant data...")
        for i in range(NUM_MERCHANTS):
            # Most merchants are low risk, few are high risk
            fraud_rate = np.random.beta(1, 50) if random.random() < 0.95 else np.random.beta(2, 5)
            
            merchant_id = f"merch_{i:06d}"
            merchant = {
                "merchant_id": merchant_id,
                "name": fake.company(),
                "category": random.choice(MERCHANT_CATEGORIES),
                "mcc": f"{random.randint(1000, 9999)}",  # Merchant Category Code
                "website": fake.url() if random.random() < 0.7 else None,
                "country": random.choice(COUNTRIES),
                "risk_score": round(np.random.beta(1, 10) * 100) / 100,  # Mostly low risk
                "fraud_rate": round(fraud_rate, 4),
                "avg_transaction_amount": round(np.random.lognormal(4, 0.8), 2),
                "is_high_risk": fraud_rate > 0.05  # High risk if fraud rate > 5%
            }
            self.merchants.append(merchant)
        
        # Save to file
        df_merchants = pd.DataFrame(self.merchants)
        df_merchants.to_csv(f"{OUTPUT_DIR}/merchants.csv", index=False)
        print(f"Generated {len(self.merchants)} merchants")
        
    def generate_locations(self):
        """Generate synthetic location data"""
        print("Generating location data...")
        for i in range(NUM_LOCATIONS):
            location_id = f"loc_{i:06d}"
            country = random.choice(COUNTRIES)
            
            # Generate coordinates based on country
            if country == "US":
                lat = random.uniform(24, 49)
                lon = random.uniform(-125, -66)
            elif country == "CA":
                lat = random.uniform(43, 57)
                lon = random.uniform(-130, -55)
            elif country == "UK":
                lat = random.uniform(50, 58)
                lon = random.uniform(-8, 2)
            else:
                lat = random.uniform(-60, 70)
                lon = random.uniform(-180, 180)
            
            location = {
                "location_id": location_id,
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "country": country,
                "city": fake.city(),
                "postal_code": fake.postcode(),
                "address": fake.street_address(),
                "timezone": fake.timezone(),
                "risk_score": round(np.random.beta(1, 10) * 100) / 100  # Mostly low risk
            }
            self.locations.append(location)
        
        # Save to file
        df_locations = pd.DataFrame(self.locations)
        df_locations.to_csv(f"{OUTPUT_DIR}/locations.csv", index=False)
        print(f"Generated {len(self.locations)} locations")
        
    def generate_devices(self):
        """Generate synthetic device and IP data"""
        print("Generating device and IP data...")
        for i in range(NUM_DEVICES):
            device_id = f"device_{i:06d}"
            
            # Generate IP address
            ip = fake.ipv4()
            ip_data = {
                "ip": ip,
                "country": random.choice(COUNTRIES),
                "city": fake.city(),
                "isp": fake.company(),
                "is_proxy": random.random() < 0.05,  # 5% are proxies
                "is_vpn": random.random() < 0.03,    # 3% are VPNs
                "is_tor": random.random() < 0.01,    # 1% are Tor
                "risk_score": round(np.random.beta(1, 10) * 100) / 100  # Mostly low risk
            }
            
            # Increase risk score for suspicious IPs
            if ip_data["is_proxy"] or ip_data["is_vpn"] or ip_data["is_tor"]:
                ip_data["risk_score"] = round(np.random.beta(5, 2) * 100) / 100  # Higher risk
            
            self.ip_addresses.append(ip_data)
            
            # Generate device
            device = {
                "device_id": device_id,
                "device_type": random.choice(DEVICE_TYPES),
                "browser": fake.user_agent() if "desktop" in device_id else fake.android_platform_token(),
                "operating_system": self._get_os_from_device_type(device_id),
                "fingerprint": str(uuid.uuid4()),
                "is_mobile": "mobile" in device_id or "tablet" in device_id,
                "is_known_device": random.random() < 0.9,  # 90% are known devices
                "risk_score": round(np.random.beta(1, 10) * 100) / 100,  # Mostly low risk
                "ip": ip
            }
            self.devices.append(device)
        
        # Save to files
        df_devices = pd.DataFrame(self.devices)
        df_devices.to_csv(f"{OUTPUT_DIR}/devices.csv", index=False)
        
        df_ips = pd.DataFrame(self.ip_addresses)
        df_ips.to_csv(f"{OUTPUT_DIR}/ip_addresses.csv", index=False)
        
        print(f"Generated {len(self.devices)} devices and {len(self.ip_addresses)} IP addresses")
    
    def _get_os_from_device_type(self, device_type):
        """Helper to get OS from device type"""
        if "android" in device_type:
            return "Android " + str(random.randint(8, 13))
        elif "ios" in device_type:
            return "iOS " + str(random.randint(12, 16))
        elif "windows" in device_type:
            return "Windows " + random.choice(["10", "11"])
        elif "mac" in device_type:
            return "macOS " + random.choice(["10.15", "11.0", "12.0", "13.0"])
        elif "linux" in device_type:
            return "Linux " + random.choice(["Ubuntu", "Debian", "Fedora", "CentOS"])
        else:
            return "Unknown"
    
    def generate_normal_transactions(self):
        """Generate normal (non-fraudulent) transactions"""
        print("Generating normal transactions...")
        
        # Get active cards
        active_cards = [card for card in self.cards if card["is_active"] and not card["is_blocked"]]
        
        for i in range(NUM_NORMAL_TRANSACTIONS):
            # Select a random card
            card = random.choice(active_cards)
            user = next(user for user in self.users if user["user_id"] == card["user_id"])
            
            # Determine transaction timestamp
            now = datetime.datetime.now()
            tx_timestamp = fake.date_time_between(
                start_date=max(
                    datetime.datetime.fromisoformat(card["issue_date"]),
                    now - datetime.timedelta(days=90)
                ),
                end_date=now
            )
            
            # Select merchant (with bias towards user's typical merchants)
            if random.random() < 0.7 and user["typical_merchants"]:
                merchant_idx = random.choice(user["typical_merchants"])
            else:
                merchant_idx = random.randint(0, NUM_MERCHANTS - 1)
            merchant = self.merchants[merchant_idx]
            
            # Select location (with bias towards user's typical locations)
            if random.random() < 0.8 and user["typical_locations"]:
                location_idx = random.choice(user["typical_locations"])
            else:
                location_idx = random.randint(0, NUM_LOCATIONS - 1)
            location = self.locations[location_idx]
            
            # Determine if online transaction
            is_online = random.random() < 0.4  # 40% are online
            
            # Select device and IP for online transactions
            device = None
            if is_online:
                device = random.choice(self.devices)
            
            # Determine transaction amount (based on user's typical amount with some variation)
            typical_amount = user["typical_transaction_amount"]
            amount = round(typical_amount * np.random.lognormal(0, 0.5), 2)
            
            # Ensure amount doesn't exceed available credit
            if amount > card["available_credit"]:
                amount = round(card["available_credit"] * random.uniform(0.1, 0.9), 2)
            
            # Create transaction
            transaction_id = f"tx_{len(self.normal_transactions) + len(self.fraudulent_transactions):08d}"
            transaction = {
                "transaction_id": transaction_id,
                "timestamp": tx_timestamp.isoformat(),
                "card_id": card["card_id"],
                "user_id": user["user_id"],
                "merchant_id": merchant["merchant_id"],
                "location_id": location["location_id"],
                "device_id": device["device_id"] if device else None,
                "amount": amount,
                "currency": "USD",  # Simplified to USD for now
                "status": "approved",
                "transaction_type": random.choice(TRANSACTION_TYPES),
                "payment_method": self._get_payment_method(is_online),
                "is_online": is_online,
                "is_fraudulent": False,
                "fraud_score": round(np.random.beta(1, 20) * 100) / 100,  # Very low fraud scores
                "mcc": merchant["mcc"],
                "auth_code": f"{random.randint(100000, 999999)}",
                "response_code": "00",  # Approved
                "is_declined": False,
                "decline_reason": None
            }
            
            self.normal_transactions.append(transaction)
            
            # Update card available credit
            if transaction["transaction_type"] == "purchase":
                card["available_credit"] = max(0, card["available_credit"] - amount)
        
        # Save to file
        df_normal = pd.DataFrame(self.normal_transactions)
        df_normal.to_csv(f"{OUTPUT_DIR}/normal_transactions.csv", index=False)
        print(f"Generated {len(self.normal_transactions)} normal transactions")
    
    def _get_payment_method(self, is_online):
        """Helper to get payment method based on transaction type"""
        if is_online:
            return "online"
        else:
            return random.choice(["chip", "swipe", "contactless", "manual_entry"])
    
    def generate_fraudulent_transactions(self):
        """Generate fraudulent transactions with various fraud patterns"""
        print("Generating fraudulent transactions...")
        
        # Get all cards
        all_cards = self.cards.copy()
        
        for i in range(NUM_FRAUDULENT_TRANSACTIONS):
            # Select a fraud pattern
            fraud_pattern = random.choice(FRAUD_PATTERNS)
            
            # Select a card (may be active or inactive depending on fraud pattern)
            if fraud_pattern == "card_testing":
                # Prefer active cards for card testing
                card = random.choice([c for c in all_cards if c["is_active"] and not c["is_blocked"]])
            elif fraud_pattern == "identity_theft":
                # Any card can be subject to identity theft
                card = random.choice(all_cards)
            else:
                # Other patterns typically target active cards
                card = random.choice([c for c in all_cards if c["is_active"]])
            
            user = next(user for user in self.users if user["user_id"] == card["user_id"])
            
            # Determine transaction timestamp
            now = datetime.datetime.now()
            tx_timestamp = fake.date_time_between(
                start_date=now - datetime.timedelta(days=30),
                end_date=now
            )
            
            # Apply fraud pattern-specific logic
            transaction = self._create_fraudulent_transaction(
                fraud_pattern, card, user, tx_timestamp, i
            )
            
            self.fraudulent_transactions.append(transaction)
        
        # Save to file
        df_fraud = pd.DataFrame(self.fraudulent_transactions)
        df_fraud.to_csv(f"{OUTPUT_DIR}/fraudulent_transactions.csv", index=False)
        print(f"Generated {len(self.fraudulent_transactions)} fraudulent transactions")
    
    def _create_fraudulent_transaction(self, fraud_pattern, card, user, timestamp, index):
        """Create a fraudulent transaction based on the specified pattern"""
        transaction_id = f"tx_f{index:08d}"
        
        # Default values
        merchant = random.choice(self.merchants)
        location = random.choice(self.locations)
        is_online = random.random() < 0.6  # Fraud is more common online
        device = random.choice(self.devices) if is_online else None
        amount = round(np.random.lognormal(4, 1), 2)
        status = "approved"
        is_declined = False
        decline_reason = None
        
        # Pattern-specific adjustments
        if fraud_pattern == "card_testing":
            # Small amounts at various merchants
            amount = round(random.uniform(0.5, 10), 2)
            merchant = random.choice([m for m in self.merchants if m["category"] in ["Online Retail", "Digital Goods"]])
            is_online = True
            device = random.choice(self.devices)
            
        elif fraud_pattern == "identity_theft":
            # Unusual location and spending pattern
            location = random.choice([l for l in self.locations if l["country"] not in ["US", "CA"]])
            amount = round(user["typical_transaction_amount"] * random.uniform(2, 5), 2)
            
        elif fraud_pattern == "account_takeover":
            # Sudden change in behavior
            is_online = True
            device = random.choice([d for d in self.devices if d["is_known_device"] == False])
            merchant = random.choice([m for m in self.merchants if m["category"] not in ["Grocery", "Restaurant"]])
            
        elif fraud_pattern == "card_not_present":
            # Online transactions with suspicious patterns
            is_online = True
            device = random.choice([d for d in self.devices if d["risk_score"] > 0.5])
            
        elif fraud_pattern == "merchant_fraud":
            # Transactions at high-risk merchants
            merchant = random.choice([m for m in self.merchants if m["is_high_risk"]])
            
        elif fraud_pattern == "application_fraud":
            # New account with unusual activity
            amount = round(random.uniform(500, 5000), 2)
            
        elif fraud_pattern == "transaction_laundering":
            # Complex chain of transactions
            merchant = random.choice([m for m in self.merchants if m["category"] in ["Money Transfer", "Financial Services"]])
            
        elif fraud_pattern == "velocity_abuse":
            # Many transactions in short time
            amount = round(random.uniform(50, 500), 2)
            
        elif fraud_pattern == "location_anomaly":
            # Transactions in multiple distant locations
            # Find a location far from user's typical locations
            if user["typical_locations"]:
                typical_loc = self.locations[random.choice(user["typical_locations"])]
                far_locations = []
                for loc in self.locations:
                    dist = self._haversine(
                        typical_loc["latitude"], typical_loc["longitude"],
                        loc["latitude"], loc["longitude"]
                    )
                    if dist > 5000:  # More than 5000 km away
                        far_locations.append(loc)
                
                if far_locations:
                    location = random.choice(far_locations)
            
        elif fraud_pattern == "amount_anomaly":
            # Unusually large transaction amounts
            amount = round(user["typical_transaction_amount"] * random.uniform(10, 50), 2)
        
        # Determine if transaction is declined (some fraud is caught)
        if random.random() < 0.3:  # 30% of fraud is caught
            status = "declined"
            is_declined = True
            decline_reason = random.choice([
                "suspected_fraud", "insufficient_funds", "invalid_card", 
                "security_violation", "exceeds_limit"
            ])
        
        # Create the transaction
        transaction = {
            "transaction_id": transaction_id,
            "timestamp": timestamp.isoformat(),
            "card_id": card["card_id"],
            "user_id": user["user_id"],
            "merchant_id": merchant["merchant_id"],
            "location_id": location["location_id"],
            "device_id": device["device_id"] if device else None,
            "amount": amount,
            "currency": "USD",
            "status": status,
            "transaction_type": "purchase",  # Most fraud is purchases
            "payment_method": self._get_payment_method(is_online),
            "is_online": is_online,
            "is_fraudulent": True,
            "fraud_pattern": fraud_pattern,
            "fraud_score": round(np.random.beta(5, 2) * 100) / 100,  # Higher fraud scores
            "mcc": merchant["mcc"],
            "auth_code": f"{random.randint(100000, 999999)}" if not is_declined else None,
            "response_code": "00" if not is_declined else "05",
            "is_declined": is_declined,
            "decline_reason": decline_reason
        }
        
        return transaction
    
    def _haversine(self, lat1, lon1, lat2, lon2):
        """Calculate the great circle distance between two points on the earth"""
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r
    
    def combine_transactions(self):
        """Combine normal and fraudulent transactions into a single dataset"""
        print("Combining transactions...")
        
        # Combine transactions
        all_transactions = self.normal_transactions + self.fraudulent_transactions
        
        # Sort by timestamp
        all_transactions.sort(key=lambda x: x["timestamp"])
        
        # Save to file
        df_all = pd.DataFrame(all_transactions)
        df_all.to_csv(f"{OUTPUT_DIR}/all_transactions.csv", index=False)
        
        # Create a JSON version for easier loading into Neo4j
        with open(f"{OUTPUT_DIR}/all_transactions.json", 'w') as f:
            json.dump(all_transactions, f)
        
        print(f"Combined {len(all_transactions)} transactions")
        
        # Create a summary of fraud patterns
        fraud_summary = {}
        for tx in self.fraudulent_transactions:
            pattern = tx["fraud_pattern"]
            if pattern in fraud_summary:
                fraud_summary[pattern] += 1
            else:
                fraud_summary[pattern] = 1
        
        print("\nFraud Pattern Summary:")
        for pattern, count in fraud_summary.items():
            print(f"  {pattern}: {count} transactions")
    
    def generate_all_data(self):
        """Generate all datasets"""
        self.generate_users()
        self.generate_cards()
        self.generate_merchants()
        self.generate_locations()
        self.generate_devices()
        self.generate_normal_transactions()
        self.generate_fraudulent_transactions()
        self.combine_transactions()
        
        print("\nData generation complete!")
        print(f"All files saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate data
    generator = DataGenerator()
    generator.generate_all_data()
