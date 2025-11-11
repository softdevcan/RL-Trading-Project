"""Item schemas for request/response validation"""

from pydantic import BaseModel, Field


class ItemBase(BaseModel):
    """Base Item schema"""
    name: str = Field(..., min_length=1, max_length=100, description="Item name")
    description: str | None = Field(None, max_length=500, description="Item description")
    price: float = Field(..., gt=0, description="Item price (must be positive)")
    quantity: int = Field(..., ge=0, description="Item quantity (must be non-negative)")


class ItemCreate(ItemBase):
    """Schema for creating a new item"""
    pass


class ItemUpdate(BaseModel):
    """Schema for updating an item"""
    name: str | None = Field(None, min_length=1, max_length=100)
    description: str | None = Field(None, max_length=500)
    price: float | None = Field(None, gt=0)
    quantity: int | None = Field(None, ge=0)


class ItemResponse(ItemBase):
    """Schema for item response"""
    id: int = Field(..., description="Item ID")

    class Config:
        from_attributes = True
