# FastAPI ML Deployment Project

This project implements a machine learning model for Census income prediction and deploys it as a RESTful API using FastAPI on Render.

## Project Structure

```
project/
├── starter/                # ML model package
│   ├── ml/
│   │   ├── data.py         # Data preprocessing functions
│   │   └── model.py        # Model training and inference functions
│   └── train_model.py      # Training script entry point
├── data/
│   └── census.csv          # Census dataset (needs cleaning)
├── main.py                 # FastAPI application (to be implemented)
├── docs/                   # German documentation (detailed, beginner-friendly)
├── requirements.txt        # Python dependencies
├── setup.py                # Package setup configuration
├── sanitycheck.py          # Test validation script
└── model_card_template.md  # Template for model documentation
```

## Environment Setup

### Using pip and venv (Recommended)
```bash
# Ensure Python 3.13 is installed
python3.13 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Development Tasks

### 1. Data Preparation
- [ ] Clean the census.csv file (remove spaces)
- [ ] Set up DVC for data versioning
- [ ] Commit cleaned data to DVC

### 2. Model Development
- [ ] Complete `train_model()` function in `starter/ml/model.py`
- [ ] Complete `inference()` function in `starter/ml/model.py`
- [ ] Write unit tests for at least 3 model functions
- [ ] Implement function for performance on data slices
- [ ] Create model card using the provided template

### 3. API Development
- [ ] Implement GET endpoint (welcome message) in `main.py`
- [ ] Implement POST endpoint (model inference) in `main.py`
- [ ] Create Pydantic model for POST body with example
- [ ] Handle column names with hyphens using Pydantic Field aliases
- [ ] Write 3 unit tests (1 GET, 2 POST)

### 4. Deployment
- [ ] Set up GitHub repository
- [ ] Configure GitHub Actions (pytest + flake8 on Python 3.13)
- [ ] Deploy to Render
- [ ] Configure environment variables on Render
- [ ] Set up DVC for data versioning (if needed)
- [ ] Write script to test live API

## Running the Application

### Local Development
```bash
# Activate virtual environment
source .venv/bin/activate

# Run the FastAPI application
uvicorn main:app --reload
```

### Testing
```bash
# Run all tests
pytest

# Run linting
flake8

# Run sanity check
python sanitycheck.py
```

## Notes

- Python version: 3.13
- Column names contain hyphens - use Pydantic Field aliases to handle them
- Model files (*.pkl, *.joblib) should be committed to GitHub for production use
- Use DVC for data versioning and model artifacts
- CI/CD pipeline should run pytest and flake8 on Python 3.13

## Deployment on Render

Render is a cloud platform that offers a free tier for small projects, making it ideal for learning and development. It provides automatic deployments from GitHub and supports FastAPI applications out of the box.

### Setting up Render

1. **Create a Render account**: Sign up at [render.com](https://render.com) (free tier available)

2. **Create a new Web Service**:
   - Connect your GitHub repository
   - Select the repository and branch
   - Choose "Web Service" as the service type

3. **Configure the service**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment**: Python 3.13
   - **Plan**: Free (for development/testing)

4. **Environment Variables** (if needed):
   - Add any required environment variables in the Render dashboard
   - For AWS/DVC access, configure `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`

5. **Automatic Deployments**:
   - Render automatically deploys on every push to the connected branch
   - You can enable "Auto-Deploy" in the service settings

### Render vs. Heroku

- **Free Tier**: Render offers a free tier that doesn't require a credit card
- **Automatic HTTPS**: SSL certificates are automatically provisioned
- **GitHub Integration**: Seamless integration with GitHub for CI/CD
- **Sleep Mode**: Free tier services sleep after 15 minutes of inactivity (wakes on first request)

### Important Notes for Render Deployment

- Use `0.0.0.0` as the host in uvicorn to bind to all interfaces
- Use `$PORT` environment variable (provided by Render) for the port
- Ensure all dependencies are listed in `requirements.txt`
- Model files should be committed to the repository or loaded from external storage (S3, etc.)

For detailed German documentation, see the `/docs` folder.

## Documentation

Detailed German documentation for beginners is available in the `/docs` folder. This documentation includes:
- Background knowledge and concepts
- Step-by-step explanations
- Detailed implementation guides
- Troubleshooting tips

## Dataset Information

The Census dataset is used for income prediction. More information can be found [here](https://archive.ics.uci.edu/ml/datasets/census+income).
