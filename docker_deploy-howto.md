# Deploying and Updating Streamlit App on Google Cloud Run

This guide explains how to **deploy, update, and optionally automate deployments** of your Streamlit app using Google Cloud Run.

---

## Updating Your App (Manual Workflow)

### 1. Update dependencies
```bash
nano requirements.txt
```

### 2. Rebuild & push image:
```bash
gcloud builds submit \
  --tag us-central1-docker.pkg.dev/second-sandbox-470608-m2/my-docker-repo/automated_test_cases_app:latest
```

### 3. Redeploy to Cloud Run::
```bash
gcloud run deploy automated-test-cases-app \
  --image us-central1-docker.pkg.dev/second-sandbox-470608-m2/my-docker-repo/automated_test_cases_app:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

```
## That’s it! Your app will be updated with the latest changes.