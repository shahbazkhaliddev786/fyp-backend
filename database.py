import ssl
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.engine.url import URL

# Define Base
Base = declarative_base()

# Your DB URL
DATABASE_URL = "postgresql+asyncpg://neondb_owner:npg_B2jeuFbW3AUY@ep-divine-sky-a4ytscdf-pooler.us-east-1.aws.neon.tech/neondb"

# SSL config
ssl_context = ssl.create_default_context()

# Create engine
engine = create_async_engine(
    DATABASE_URL,
    echo=True,
    connect_args={"ssl": ssl_context},  
)

# Async session
async_session = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

# Dependency
async def get_db():
    async with async_session() as session:
        yield session
