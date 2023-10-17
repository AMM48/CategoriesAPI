from typing import Optional
from pydantic import BaseModel

class Transaction(BaseModel):
    message: str
