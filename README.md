# BALDER - Trading Signals App

Full-stack trading application with FastAPI backend and React frontend.

## Quick Start

### Start Backend (Terminal 1)
```bash
cd backend
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Start Frontend (Terminal 2)
```bash
cd frontend
npm run dev
```

### Access App
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Passcode: `000`

## Or Use Scripts

```bash
./start-backend.sh    # Terminal 1
./start-frontend.sh   # Terminal 2
./start-all.sh        # Start both (background)
```

## Project Structure

```
trade/
├── backend/          # FastAPI Python backend
│   ├── main.py
│   ├── core/        # Trading logic
│   └── api/         # API endpoints
├── frontend/        # React TypeScript frontend  
│   └── src/
└── trend_app.py     # Original Streamlit app
```

## Troubleshooting

**Backend returns 0 values:**
1. Kill backend: `lsof -ti:8000 | xargs kill -9`
2. Clear cache: `find backend -name "__pycache__" -exec rm -rf {} +`
3. Restart backend manually

**CORS errors:**
- Make sure backend is on port 8000
- Make sure frontend is on port 5173
- Hard refresh browser: `Cmd+Shift+R`

**Port already in use:**
```bash
lsof -ti:8000 | xargs kill -9  # Backend
lsof -ti:5173 | xargs kill -9  # Frontend
```
