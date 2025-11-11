"""Item management endpoints"""

from fastapi import APIRouter, HTTPException, Path, Query
from typing import List

from app.schemas.item import ItemCreate, ItemResponse, ItemUpdate

router = APIRouter(prefix="/items", tags=["Items"])

# In-memory storage for demonstration (replace with database in production)
items_db: dict[int, dict] = {}
next_id = 1


@router.get("/", response_model=List[ItemResponse])
async def list_items(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of items to return")
):
    """List all items with pagination"""
    items_list = list(items_db.values())[skip:skip + limit]
    return items_list


@router.get("/{item_id}", response_model=ItemResponse)
async def get_item(
    item_id: int = Path(..., gt=0, description="Item ID")
):
    """Get a specific item by ID"""
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return items_db[item_id]


@router.post("/", response_model=ItemResponse, status_code=201)
async def create_item(item: ItemCreate):
    """Create a new item"""
    global next_id

    item_dict = item.model_dump()
    item_dict["id"] = next_id
    items_db[next_id] = item_dict

    response = ItemResponse(**item_dict)
    next_id += 1

    return response


@router.put("/{item_id}", response_model=ItemResponse)
async def update_item(
    item_id: int = Path(..., gt=0, description="Item ID"),
    item_update: ItemUpdate = ...
):
    """Update an existing item"""
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")

    stored_item = items_db[item_id]
    update_data = item_update.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        stored_item[field] = value

    return ItemResponse(**stored_item)


@router.delete("/{item_id}", status_code=204)
async def delete_item(
    item_id: int = Path(..., gt=0, description="Item ID")
):
    """Delete an item"""
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")

    del items_db[item_id]
    return None
