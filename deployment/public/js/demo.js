// Credit Card Fraud Detection Demo - Main JavaScript
// This file contains the JavaScript code for the demo website

// Sample data for merchants, locations, and users
const merchants = [
    { id: 1, name: "Electronics Megastore", category: "Electronics" },
    { id: 2, name: "Luxury Boutique", category: "Retail" },
    { id: 3, name: "Online Marketplace", category: "E-commerce" },
    { id: 4, name: "Travel Agency", category: "Travel" },
    { id: 5, name: "Grocery Store", category: "Grocery" },
    { id: 6, name: "Gas Station", category: "Fuel" },
    { id: 7, name: "Restaurant", category: "Food" },
    { id: 8, name: "Department Store", category: "Retail" },
    { id: 9, name: "Online Gaming", category: "Entertainment" },
    { id: 10, name: "Jewelry Store", category: "Luxury" }
];

const locations = [
    { id: 1, city: "New York", country: "USA" },
    { id: 2, city: "London", country: "UK" },
    { id: 3, city: "Tokyo", country: "Japan" },
    { id: 4, city: "Paris", country: "France" },
    { id: 5, city: "Sydney", country: "Australia" },
    { id: 6, city: "Berlin", country: "Germany" },
    { id: 7, city: "Toronto", country: "Canada" },
    { id: 8, city: "Moscow", country: "Russia" },
    { id: 9, city: "Dubai", country: "UAE" },
    { id: 10, city: "Singapore", country: "Singapore" }
];

const users = [
    { id: 1, name: "John Smith" },
    { id: 2, name: "Emma Johnson" },
    { id: 3, name: "Michael Brown" },
    { id: 4, name: "Sophia Davis" },
    { id: 5, name: "William Wilson" },
    { id: 6, name: "Olivia Martinez" },
    { id: 7, name: "James Taylor" },
    { id: 8, name: "Ava Anderson" },
    { id: 9, name: "Alexander Thomas" },
    { id: 10, name: "Charlotte Jackson" }
];

// Fraud patterns
const fraudPatterns = [
    "card_testing",
    "identity_theft",
    "account_takeover",
    "card_not_present",
    "merchant_fraud",
    "velocity_abuse",
    "location_anomaly",
    "amount_anomaly"
];

// Agent types
const agentTypes = [
    "Transaction Analysis Agent",
    "Pattern Detection Agent",
    "User Profile Agent",
    "Merchant Risk Agent",
    "Investigation Agent",
    "Decision Agent",
    "Feedback Collection Agent",
    "Learning Agent"
];

// Transaction counter
let transactionCounter = 0;
let fraudCounter = 0;
let totalFraudScore = 0;
let fraudPatternCounts = {};

// Initialize fraud pattern counts
fraudPatterns.forEach(pattern => {
    fraudPatternCounts[pattern] = 0;
});

// Initialize charts
let fraudPatternChart;
let fraudScoreChart;

// Initialize charts when the page loads
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    
    // Enable/disable fraud pattern select based on transaction type
    document.getElementById('transactionType').addEventListener('change', function() {
        const fraudPatternSelect = document.getElementById('fraudPattern');
        fraudPatternSelect.disabled = this.value !== 'fraudulent';
    });
    
    // Generate transaction button click handler
    document.getElementById('generateTransaction').addEventListener('click', function() {
        generateTransaction();
    });
    
    // Generate some initial transactions
    for (let i = 0; i < 5; i++) {
        setTimeout(() => {
            generateTransaction(true);
        }, i * 1000);
    }
});

function initializeCharts() {
    // Fraud Pattern Chart
    const fraudPatternCtx = document.getElementById('fraudPatternChart').getContext('2d');
    fraudPatternChart = new Chart(fraudPatternCtx, {
        type: 'bar',
        data: {
            labels: fraudPatterns.map(pattern => formatFraudPattern(pattern)),
            datasets: [{
                label: 'Number of Frauds',
                data: fraudPatterns.map(pattern => fraudPatternCounts[pattern]),
                backgroundColor: 'rgba(220, 53, 69, 0.7)',
                borderColor: 'rgba(220, 53, 69, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        precision: 0
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
    
    // Fraud Score Chart
    const fraudScoreCtx = document.getElementById('fraudScoreChart').getContext('2d');
    fraudScoreChart = new Chart(fraudScoreCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Fraud Score',
                data: [],
                backgroundColor: 'rgba(0, 102, 204, 0.2)',
                borderColor: 'rgba(0, 102, 204, 1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
}

function generateTransaction(isAutomatic = false) {
    // Get transaction type
    let transactionType;
    if (isAutomatic) {
        // Random transaction type with 70% normal, 20% suspicious, 10% fraudulent
        const rand = Math.random();
        if (rand < 0.7) {
            transactionType = 'normal';
        } else if (rand < 0.9) {
            transactionType = 'suspicious';
        } else {
            transactionType = 'fraudulent';
        }
    } else {
        transactionType = document.getElementById('transactionType').value;
    }
    
    // Generate transaction data
    const transactionId = `TX${String(Date.now()).slice(-8)}`;
    const amount = transactionType === 'fraudulent' ? 
        Math.floor(Math.random() * 9000) + 1000 : // Higher amounts for fraudulent
        Math.floor(Math.random() * 500) + 50; // Lower amounts for normal
    
    const merchantIndex = Math.floor(Math.random() * merchants.length);
    const merchant = merchants[merchantIndex];
    
    const locationIndex = Math.floor(Math.random() * locations.length);
    const location = locations[locationIndex];
    
    const userIndex = Math.floor(Math.random() * users.length);
    const user = users[userIndex];
    
    const isOnline = Math.random() > 0.5;
    
    // Generate fraud score based on transaction type
    let fraudScore;
    if (transactionType === 'normal') {
        fraudScore = Math.random() * 0.2; // 0.0 - 0.2
    } else if (transactionType === 'suspicious') {
        fraudScore = 0.3 + Math.random() * 0.4; // 0.3 - 0.7
    } else {
        fraudScore = 0.7 + Math.random() * 0.3; // 0.7 - 1.0
    }
    
    // Get fraud pattern if fraudulent
    let fraudPattern = null;
    if (transactionType === 'fraudulent') {
        if (isAutomatic) {
            const patternIndex = Math.floor(Math.random() * fraudPatterns.length);
            fraudPattern = fraudPatterns[patternIndex];
        } else {
            fraudPattern = document.getElementById('fraudPattern').value;
        }
        
        // Update fraud pattern counts
        fraudPatternCounts[fraudPattern]++;
    }
    
    // Create transaction object
    const transaction = {
        id: transactionId,
        timestamp: new Date(),
        amount: amount,
        merchant: merchant,
        location: location,
        user: user,
        isOnline: isOnline,
        fraudScore: fraudScore,
        type: transactionType,
        fraudPattern: fraudPattern
    };
    
    // Add transaction to feed
    addTransactionToFeed(transaction);
    
    // Update metrics
    updateMetrics(transaction);
    
    // Update charts
    updateCharts(transaction);
    
    // Simulate agent activity
    simulateAgentActivity(transaction);
    
    return transaction;
}

function addTransactionToFeed(transaction) {
    const transactionFeed = document.getElementById('transactionFeed');
    
    // Create transaction element
    const transactionElement = document.createElement('div');
    transactionElement.className = `transaction-item transaction-${transaction.type}`;
    
    // Format timestamp
    const timestamp = transaction.timestamp.toLocaleTimeString();
    
    // Create transaction content
    let transactionContent = `
        <div class="transaction-status status-${transaction.type}">${formatTransactionType(transaction.type)}</div>
        <div class="transaction-amount">$${transaction.amount.toFixed(2)}</div>
        <div class="transaction-merchant">${transaction.merchant.name}</div>
        <div class="transaction-details">
            <span class="transaction-time">${timestamp}</span> | 
            <span>${transaction.user.name}</span> | 
            <span>${transaction.location.city}, ${transaction.location.country}</span> | 
            <span>${transaction.isOnline ? 'Online' : 'In-store'}</span>
        </div>
        <div class="mt-2">
            <strong>Fraud Score:</strong> ${transaction.fraudScore.toFixed(2)}
        </div>
    `;
    
    // Add fraud pattern if fraudulent
    if (transaction.type === 'fraudulent' && transaction.fraudPattern) {
        transactionContent += `
            <div class="mt-1">
                <strong>Fraud Pattern:</strong> ${formatFraudPattern(transaction.fraudPattern)}
            </div>
        `;
    }
    
    transactionElement.innerHTML = transactionContent;
    
    // Add to feed (at the beginning)
    transactionFeed.insertBefore(transactionElement, transactionFeed.firstChild);
    
    // Limit to 10 transactions in the feed
    if (transactionFeed.children.length > 10) {
        transactionFeed.removeChild(transactionFeed.lastChild);
    }
}

function updateMetrics(transaction) {
    // Update transaction counter
    transactionCounter++;
    document.getElementById('totalTransactions').textContent = transactionCounter;
    
    // Update fraud counter if fraudulent
    if (transaction.type === 'fraudulent') {
        fraudCounter++;
        document.getElementById('fraudulentTransactions').textContent = fraudCounter;
    }
    
    // Update fraud rate
    const fraudRate = (fraudCounter / transactionCounter * 100).toFixed(1);
    document.getElementById('fraudRate').textContent = `${fraudRate}%`;
    
    // Update average fraud score
    totalFraudScore += transaction.fraudScore;
    const avgFraudScore = (totalFraudScore / transactionCounter).toFixed(2);
    document.getElementById('avgFraudScore').textContent = avgFraudScore;
}

function updateCharts(transaction) {
    // Update fraud pattern chart
    fraudPatternChart.data.datasets[0].data = fraudPatterns.map(pattern => fraudPatternCounts[pattern]);
    fraudPatternChart.update();
    
    // Update fraud score chart
    if (fraudScoreChart.data.labels.length >= 20) {
        fraudScoreChart.data.labels.shift();
        fraudScoreChart.data.datasets[0].data.shift();
    }
    
    fraudScoreChart.data.labels.push(transaction.id);
    fraudScoreChart.data.datasets[0].data.push(transaction.fraudScore);
    fraudScoreChart.update();
}

function simulateAgentActivity(transaction) {
    const agentActivity = document.getElementById('agentActivity');
    
    // Determine which agents to show based on transaction type
    let agentsToShow;
    if (transaction.type === 'normal') {
        agentsToShow = [0, 2]; // Transaction Analysis, User Profile
    } else if (transaction.type === 'suspicious') {
        agentsToShow = [0, 1, 2, 3, 5]; // All except Investigation and Learning
    } else {
        agentsToShow = [0, 1, 2, 3, 4, 5, 6, 7]; // All agents
    }
    
    // Create agent activity log
    const activityLog = document.createElement('div');
    activityLog.className = 'card mb-3';
    
    let activityContent = `
        <div class="card-header">
            Transaction ${transaction.id} - ${formatTransactionType(transaction.type)}
        </div>
        <div class="card-body">
            <h6>Agent Activity Log</h6>
            <div class="mt-3">
    `;
    
    // Add agent activities
    agentsToShow.forEach(agentIndex => {
        const agentType = agentTypes[agentIndex];
        const processingTime = (Math.random() * 0.5 + 0.1).toFixed(2);
        
        let agentAction;
        let agentResult;
        
        switch (agentIndex) {
            case 0: // Transaction Analysis Agent
                agentAction = "Analyzing transaction";
                agentResult = `Initial fraud score: ${transaction.fraudScore.toFixed(2)}`;
                break;
            case 1: // Pattern Detection Agent
                agentAction = "Detecting patterns";
                if (transaction.type === 'fraudulent') {
                    agentResult = `Pattern detected: ${formatFraudPattern(transaction.fraudPattern)}`;
                } else if (transaction.type === 'suspicious') {
                    agentResult = "Suspicious activity detected";
                } else {
                    agentResult = "No patterns detected";
                }
                break;
            case 2: // User Profile Agent
                agentAction = "Analyzing user behavior";
                agentResult = "User profile analyzed";
                break;
            case 3: // Merchant Risk Agent
                agentAction = "Assessing merchant risk";
                agentResult = `Merchant risk level: ${(Math.random() * 0.5).toFixed(2)}`;
                break;
            case 4: // Investigation Agent
                agentAction = "Investigating transaction";
                agentResult = "Investigation complete";
                break;
            case 5: // Decision Agent
                agentAction = "Making final determination";
                agentResult = `Decision: ${transaction.type === 'fraudulent' ? 'Reject' : transaction.type === 'suspicious' ? 'Flag for review' : 'Approve'}`;
                break;
            case 6: // Feedback Collection Agent
                agentAction = "Collecting feedback";
                agentResult = "Feedback recorded";
                break;
            case 7: // Learning Agent
                agentAction = "Updating models";
                agentResult = "Models updated";
                break;
        }
        
        activityContent += `
            <div class="d-flex align-items-start mb-2">
                <div class="me-3">
                    <span class="badge bg-primary">${agentType}</span>
                </div>
                <div>
                    <div><strong>${agentAction}</strong> (${processingTime}s)</div>
                    <div class="text-muted">${agentResult}</div>
                </div>
            </div>
        `;
    });
    
    activityContent += `
            </div>
        </div>
    `;
    
    activityLog.innerHTML = activityContent;
    
    // Add to agent activity (at the beginning)
    agentActivity.insertBefore(activityLog, agentActivity.firstChild);
    
    // Limit to 5 activity logs
    if (agentActivity.children.length > 5) {
        agentActivity.removeChild(agentActivity.lastChild);
    }
}

function formatTransactionType(type) {
    switch (type) {
        case 'normal':
            return 'Normal';
        case 'suspicious':
            return 'Suspicious';
        case 'fraudulent':
            return 'Fraud';
        default:
            return type;
    }
}

function formatFraudPattern(pattern) {
    return pattern.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
}

// Add automatic transaction generation every 5 seconds
setInterval(() => {
    generateTransaction(true);
}, 5000);
