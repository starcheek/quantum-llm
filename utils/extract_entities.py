import hashlib, json
from typing import List, Optional

from pydantic import BaseModel, Field
from openai import OpenAI


class Concept(BaseModel):
    name: str = Field(description="Technical concept or term")
    importance: str = Field(description="primary, secondary, mentioned")


class Relationship(BaseModel):
    concept_a: str
    concept_b: str
    relationship_type: str = Field(description="relates_to | enables | improves | solves")
    description: Optional[str] = None


class PaperExtraction(BaseModel):
    concepts: List[Concept] = []
    relationships: List[Relationship] = []


EXTRACTION_SYSTEM = (
    "You are a scientific paper analyzer. Extract concise, structured concepts "
    "and relationships from the given title and abstract. Use simple concept names."
)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_paper_entities(title: str, abstract: str, *, api_key: str, model: str = "gpt-4o-mini") -> PaperExtraction:
    client = OpenAI(api_key=api_key)

    content = f"Title: {title}\n\nAbstract: {abstract}"

    # Use JSON mode for robust structured output
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        temperature=0.1,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {
                "role": "user",
                "content": (
                    "Extract JSON with keys: concepts (list of {name, importance}), "
                    "relationships (list of {concept_a, concept_b, relationship_type, description}).\n\n"
                    + content
                ),
            },
        ],
    )

    raw = resp.choices[0].message.content
    try:
        data = json.loads(raw)
    except Exception:
        data = {"concepts": [], "relationships": []}

    return PaperExtraction.model_validate(data)


def get_cache_key(doi: str, title: str, abstract: str) -> str:
    # Include title+abstract so updates invalidate cache
    return _hash_text(doi + "|" + title + "|" + abstract)


def load_cache(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(path: str, cache: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


