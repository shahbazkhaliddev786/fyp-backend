# .\venv\Scripts\activate
# pip install -r requirements.txt


from fastapi import FastAPI
from sqlalchemy import text 
from database import engine
from routes.predict_crypto import router as crypto_router
from routes.predict_stocks import router as stocks_router
from routes.auth_api import router as auth_router
import logging
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()


app.include_router(crypto_router)
app.include_router(stocks_router)
app.include_router(auth_router)

# Apply CORS middleware to handle cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

@app.on_event("startup")
async def startup():
    # Run table creation
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1")) 
        print("✅ Successfully connected to the database.")
    except Exception as e:
        print("❌ Database connection failed:", e)

@app.get("/")
async def root():
    return {"message": "FastAPI with Neon DB connected!"}


