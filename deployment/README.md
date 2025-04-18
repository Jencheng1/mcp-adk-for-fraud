# Credit Card Fraud Detection System Deployment Guide

This guide provides instructions for deploying the Credit Card Fraud Detection Demo website permanently using GitHub and AWS Amplify.

## Prerequisites

1. A GitHub account
2. An AWS account with access to AWS Amplify
3. The complete code repository for the Credit Card Fraud Detection Demo

## Deployment Steps

### Step 1: Create a GitHub Repository

1. Log in to your GitHub account
2. Click on the "+" icon in the top-right corner and select "New repository"
3. Name your repository (e.g., "credit-card-fraud-detection-demo")
4. Choose whether to make it public or private
5. Click "Create repository"

### Step 2: Push the Code to GitHub

1. Initialize a Git repository in the project directory:
   ```bash
   cd /path/to/credit_card_fraud_detection/deployment
   git init
   ```

2. Add all files to the repository:
   ```bash
   git add .
   ```

3. Commit the files:
   ```bash
   git commit -m "Initial commit"
   ```

4. Add your GitHub repository as a remote:
   ```bash
   git remote add origin https://github.com/your-username/credit-card-fraud-detection-demo.git
   ```

5. Push the code to GitHub:
   ```bash
   git push -u origin main
   ```

### Step 3: Deploy with AWS Amplify

1. Log in to the AWS Management Console
2. Navigate to AWS Amplify
3. Click "New app" and select "Host web app"
4. Choose GitHub as the repository source and connect your GitHub account
5. Select the repository you created in Step 1
6. Select the branch you want to deploy (usually "main")
7. Configure build settings:
   - Build command: `npm run build` (if using a build process) or leave empty for static site
   - Output directory: `public` (this is where our static files are located)
8. Click "Next" and then "Save and deploy"

AWS Amplify will now deploy your website and provide you with a URL where it can be accessed.

### Step 4: Configure Custom Domain (Optional)

1. In the AWS Amplify console, select your app
2. Go to "Domain management"
3. Click "Add domain"
4. Enter your domain name and follow the instructions to configure DNS settings

## Continuous Deployment

The GitHub Actions workflow file (`.github/workflows/deploy.yml`) in the repository is configured to automatically deploy changes to AWS Amplify whenever you push to the main branch.

To make updates to the website:

1. Make your changes locally
2. Commit the changes:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```
3. Push to GitHub:
   ```bash
   git push origin main
   ```

The changes will be automatically deployed to AWS Amplify.

## Troubleshooting

If you encounter issues with the deployment:

1. Check the build logs in AWS Amplify console
2. Verify that your repository structure matches the expected structure
3. Ensure all file paths in HTML, CSS, and JavaScript files are correct
4. Check that all required files are included in the repository

## Support

For additional support:

- AWS Amplify documentation: https://docs.aws.amazon.com/amplify/
- GitHub documentation: https://docs.github.com/en
- Contact the development team for specific questions about the Credit Card Fraud Detection Demo

## Security Considerations

- Ensure that sensitive information is not included in the public repository
- Use environment variables in AWS Amplify for any configuration that should not be in the code
- Regularly update dependencies to address security vulnerabilities
