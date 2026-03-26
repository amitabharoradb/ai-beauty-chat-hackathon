from pydantic import BaseModel
from .. import __version__


class VersionOut(BaseModel):
    version: str

    @classmethod
    def from_metadata(cls):
        return cls(version=__version__)


class CustomerOut(BaseModel):
    id: str
    name: str
    skin_type: str
    skin_tone: str


class MessageIn(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    customer_id: str
    message: str
    history: list[MessageIn] = []


class ProductOut(BaseModel):
    id: str
    name: str
    description: str
    price: float
    key_ingredients: list[str]
    benefits: list[str]
    tags: list[str]
    category: str
