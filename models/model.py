from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: Optional[int] = Field(10, ge=1, le=50)

class PaperItem(BaseModel):
    id: str
    title: str
    abstract: Optional[str] = None
    authors: List[str] = []
    pdf_url: Optional[str] = None
    published: Optional[datetime] = None
    arxiv_url: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    count: int
    papers: List[PaperItem]
    summary: Optional[str] = None
    top_papers: Optional[List[PaperItem]] = None
    top_paper_links: Optional[List[Dict[str, str]]] = None
    display_text: Optional[str] = None
    human_readable: Optional[str] = None  

class A2APayload(BaseModel):
    invocation_id: Optional[str] = None
    sender: Optional[str] = None
    query: str
    docs: Optional[List[Dict[str, Any]]] = None 