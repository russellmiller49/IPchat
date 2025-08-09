# Bronchmonkey Deployment Guide

## 🚀 Quick Start with Docker

### Prerequisites
- Docker and Docker Compose installed
- OpenAI API key
- 8GB+ RAM recommended
- Ports 8000 and 8501 available

### 1. Clone and Setup
```bash
git clone <your-repo>
cd IP_chat2

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env
```

### 2. Build and Run with Docker
```bash
# Build and start all services
docker-compose up -d

# Load initial data
docker-compose exec bronchmonkey python3 ingestion/load_json_to_pg.py --trials-dir data/oe_final_outputs

# Build indexes
docker-compose exec bronchmonkey python3 chunking/chunker.py --trials-dir data/oe_final_outputs
docker-compose exec bronchmonkey python3 indexing/build_bm25.py
docker-compose exec bronchmonkey python3 indexing/build_faiss.py
```

### 3. Access the Application
- **Streamlit UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🌐 Cloud Deployment Options

### Option 1: Deploy to Heroku
```bash
# Install Heroku CLI
# Create Heroku app
heroku create bronchmonkey-app

# Add PostgreSQL
heroku addons:create heroku-postgresql:mini

# Set environment variables
heroku config:set OPENAI_API_KEY=your_key_here

# Deploy
git push heroku main
```

### Option 2: Deploy to AWS EC2

1. **Launch EC2 Instance**
   - Amazon Linux 2 or Ubuntu 20.04
   - t3.large or larger (4GB+ RAM)
   - Open ports: 22, 80, 443, 8000, 8501

2. **Install Docker**
```bash
sudo yum update -y
sudo yum install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user

# Install docker-compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

3. **Deploy Application**
```bash
# Clone repository
git clone <your-repo>
cd IP_chat2

# Setup environment
cp .env.example .env
# Edit .env with your configuration

# Start services
docker-compose up -d
```

4. **Setup Nginx (Optional)**
```bash
sudo yum install nginx -y
# Configure reverse proxy to port 8501
```

### Option 3: Deploy to Google Cloud Run

1. **Build and Push Image**
```bash
# Configure gcloud
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Build and push
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/bronchmonkey

# Deploy
gcloud run deploy bronchmonkey \
  --image gcr.io/YOUR_PROJECT_ID/bronchmonkey \
  --platform managed \
  --port 8501 \
  --memory 2Gi \
  --set-env-vars "OPENAI_API_KEY=your_key"
```

### Option 4: Deploy to Azure Container Instances

```bash
# Create resource group
az group create --name bronchmonkey-rg --location eastus

# Create container
az container create \
  --resource-group bronchmonkey-rg \
  --name bronchmonkey \
  --image bronchmonkey:latest \
  --dns-name-label bronchmonkey \
  --ports 8501 8000 \
  --environment-variables \
    OPENAI_API_KEY=your_key \
    DATABASE_URL=your_db_url
```

## 📊 Production Considerations

### Database Setup
For production, use a managed PostgreSQL service:
- **AWS RDS PostgreSQL**
- **Google Cloud SQL**
- **Azure Database for PostgreSQL**
- **Supabase** (free tier available)

### Environment Variables
Required for production:
```env
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://user:pass@host:5432/dbname
SECRET_KEY=<generate-random-key>
ENVIRONMENT=production
```

### Security Checklist
- [ ] Use HTTPS (SSL certificates)
- [ ] Set strong database passwords
- [ ] Rotate API keys regularly
- [ ] Implement rate limiting
- [ ] Add authentication (optional)
- [ ] Monitor usage and costs

### Performance Optimization
1. **Cache Configuration**
   - Redis for session management
   - CloudFlare for static assets

2. **Scaling**
   - Horizontal scaling for API (multiple containers)
   - Use CDN for static files
   - Database connection pooling

3. **Monitoring**
   - Application logs
   - API usage metrics
   - Error tracking (Sentry)
   - Uptime monitoring

## 🔧 Maintenance

### Backup Data
```bash
# Backup PostgreSQL
docker-compose exec postgres pg_dump -U bronchmonkey ip_rag > backup.sql

# Backup indexes
tar -czf indexes_backup.tar.gz data/index/
```

### Update Application
```bash
# Pull latest changes
git pull origin main

# Rebuild containers
docker-compose down
docker-compose build
docker-compose up -d
```

### Monitor Logs
```bash
# View all logs
docker-compose logs -f

# View specific service
docker-compose logs -f bronchmonkey
```

## 💰 Cost Estimates

### Small Deployment (1-10 users)
- **Heroku**: Free - $7/month
- **AWS EC2 t3.small**: ~$15/month
- **Google Cloud Run**: ~$10/month
- **Azure Container**: ~$20/month

### Medium Deployment (10-50 users)
- **AWS EC2 t3.medium + RDS**: ~$80/month
- **Google Cloud Platform**: ~$75/month
- **Azure**: ~$90/month

### API Costs
- **OpenAI GPT-4**: ~$0.03 per query
- **Embedding generation**: ~$0.0001 per query

## 🆘 Troubleshooting

### Common Issues

1. **"Database connection failed"**
   - Check DATABASE_URL format
   - Ensure PostgreSQL is running
   - Check network connectivity

2. **"API key invalid"**
   - Verify OPENAI_API_KEY in .env
   - Check API key permissions

3. **"Out of memory"**
   - Increase Docker memory limits
   - Use larger instance type
   - Optimize chunk size

4. **"Port already in use"**
   - Change ports in docker-compose.yml
   - Stop conflicting services

### Support
- GitHub Issues: [your-repo/issues]
- Email: support@bronchmonkey.app

## 📝 License
See LICENSE file for details.