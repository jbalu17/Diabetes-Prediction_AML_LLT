# Diabetes Prediction ML Pipeline

A complete machine learning pipeline for diabetes prediction with FastAPI deployment and Docker containerization.

## Features

- **Machine Learning**: Random Forest Classifier trained on Pima Indians Diabetes Dataset
- **REST API**: FastAPI with automatic documentation
- **Web Interface**: Interactive prediction form
- **Containerization**: Docker and Docker Compose support
- **Testing**: Comprehensive test suite with pytest

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (optional)

### Local Setup

1. **Clone and setup**
   ```bash
   git clone <your-repo>
   cd diabetes-prediction
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
