# Credit Card Fraud Detection Demo - Deployment Guide

This guide explains how to deploy the Credit Card Fraud Detection Demo website permanently using GitHub and AWS Amplify.

## Prerequisites

1. GitHub account
2. AWS account with Amplify access
3. AWS CLI configured with appropriate credentials

## Deployment Steps

### 1. Create a GitHub Repository

1. Go to GitHub and create a new repository named `credit-card-fraud-detection-demo`
2. Initialize the repository with a README file

### 2. Prepare the Demo Website for Deployment

The demo website needs to be adapted for production deployment. The main changes include:

- Separating the static website content from the backend services
- Setting up API endpoints for the backend services
- Configuring the frontend to use the deployed API endpoints

### 3. Set Up AWS Amplify

1. Log in to the AWS Management Console
2. Navigate to AWS Amplify
3. Click "New app" and select "Host web app"
4. Choose GitHub as the repository source
5. Connect to your GitHub account and select the `credit-card-fraud-detection-demo` repository
6. Configure build settings:
   ```yaml
   version: 1
   frontend:
     phases:
       build:
         commands:
           - echo "No build required for static site"
     artifacts:
       baseDirectory: /
       files:
         - '**/*'
     cache:
       paths: []
   ```
7. Click "Save and deploy"

### 4. Set Up Backend Services

For the backend services (Streamlit dashboards, Kafka, PySpark, Neo4j), we'll use AWS Elastic Beanstalk:

1. Create an Elastic Beanstalk environment for each service
2. Deploy the backend code to the appropriate environment
3. Configure the environments with the necessary dependencies
4. Set up environment variables for service connections

### 5. Configure GitHub Actions for Continuous Deployment

The repository includes a GitHub Actions workflow file (`.github/workflows/deploy.yml`) that automatically deploys changes to AWS Amplify when you push to the main branch.

To set up the workflow:

1. In your GitHub repository, go to Settings > Secrets
2. Add the following secrets:
   - `AWS_ACCESS_KEY_ID`: Your AWS access key
   - `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
   - `AMPLIFY_APP_ID`: The ID of your Amplify app (found in the Amplify console)

### 6. Update API Endpoints in the Frontend

Update the frontend code to use the deployed API endpoints:

1. Edit the JavaScript files to point to the Elastic Beanstalk URLs
2. Push the changes to GitHub
3. Amplify will automatically deploy the updated frontend

## Monitoring and Maintenance

- Monitor the application using AWS CloudWatch
- Set up alerts for any issues
- Regularly update dependencies and security patches

## Troubleshooting

If you encounter issues with the deployment:

1. Check the Amplify build logs for frontend issues
2. Check the Elastic Beanstalk logs for backend issues
3. Verify that all environment variables are correctly set
4. Ensure that the security groups allow the necessary traffic between services

## Cost Considerations

The deployment uses several AWS services that may incur costs:

- AWS Amplify for hosting the frontend
- AWS Elastic Beanstalk for hosting the backend services
- Amazon RDS or other database services
- Data transfer and storage costs

Monitor your AWS billing dashboard to keep track of costs.
