# RL Trading API

API for Reinforcement Learning Trading System built with FastAPI.

## Project Structure

```
RL-Trading-Project/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py    # Health check endpoints
│   │       └── items.py     # Item management endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py        # Application configuration
│   ├── models/              # Database models (future)
│   │   └── __init__.py
│   └── schemas/             # Pydantic schemas
│       ├── __init__.py
│       ├── health.py
│       └── item.py
├── venv/                    # Virtual environment
├── .env.example             # Environment variables example
├── .gitignore
├── main.py                  # Application entry point
├── requirements.txt
└── README.md
```

## Setup

### 1. Create virtual environment

```bash
python -m venv venv
```

### 2. Activate virtual environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

Copy `.env.example` to `.env` and modify as needed:

```bash
cp .env.example .env
```

## Running the Application

### Development mode (with auto-reload)

```bash
python main.py
```

Or:

```bash
uvicorn app.main:app --reload
```

### Production mode

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Documentation

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Available Endpoints

### Health Check
- `GET /` - Welcome message
- `GET /health` - Health check

### Items Management
- `GET /items/` - List all items (with pagination)
- `GET /items/{item_id}` - Get item by ID
- `POST /items/` - Create new item
- `PUT /items/{item_id}` - Update item
- `DELETE /items/{item_id}` - Delete item

## Development

### Project Features

- FastAPI framework with modern Python features
- Automatic API documentation (Swagger/ReDoc)
- Request/response validation with Pydantic
- CORS middleware configured
- Environment-based configuration
- Modular project structure
- Type hints throughout

### Adding New Endpoints

1. Create schema in `app/schemas/`
2. Create route in `app/api/routes/`
3. Register router in `app/main.py`

## About

Reinforcement Learning model proposal for Algorithmic Trading

## License

MIT
