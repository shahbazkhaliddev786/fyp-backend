import asyncio
from database import engine
from db_models import Base

async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Tables created successfully.")

asyncio.run(create_tables())
