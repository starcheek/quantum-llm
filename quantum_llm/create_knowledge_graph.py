import csv, json
from neo4j import GraphDatabase
from utils.extract_entities import (
    extract_paper_entities,
    get_cache_key,
    load_cache,
    save_cache,
)

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

config = load_config()

NEO4J_URI = config["neo4j"]["uri"]
NEO4J_USER = config["neo4j"]["user"]
NEO4J_PASSWORD = config["neo4j"]["password"]

CSV_PATH = config["data"]["csv_path"]

# Extraction / cache configuration
OPENAI_API_KEY = config["llm"]["openai_api_key"]
OPENAI_MODEL = config["llm"].get("openai_model", "gpt-4o-mini")
CACHE_PATH = config.get("extraction", {}).get("cache_file", "data/extraction_cache.json")
MAX_CONCEPTS = int(config.get("extraction", {}).get("max_concepts_per_paper", 7))

SCHEMA = config["neo4j"]["constraints"]["paper_doi_unique"]

# Constraint for Year uniqueness
YEAR_CONSTRAINT = """
CREATE CONSTRAINT year_unique IF NOT EXISTS 
FOR (y:Year) REQUIRE y.year IS UNIQUE
"""

# Create/update Paper node
UPSERT_PAPER = """
MERGE (p:Paper {doi: $doi})
SET  p.title = $title,
     p.abstract = $abstract,
     p.date = date($date)
"""

# Create Year node and PUBLISHED_IN relationship
CREATE_YEAR_RELATIONSHIP = """
MERGE (y:Year {year: $year})
WITH y
MATCH (p:Paper {doi: $doi})
MERGE (p)-[:PUBLISHED_IN]->(y)
"""

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session(database=config["neo4j"]["database"]) as s:
        # Create constraints
        print("Creating constraints...")
        s.run(SCHEMA)
        s.run(YEAR_CONSTRAINT)

        # load CSV
        print("Loading papers from CSV...")
        cache = load_cache(CACHE_PATH)
        with open(CSV_PATH, newline="", encoding=config["data"]["encoding"]) as f:
            reader = csv.DictReader(f)
            paper_count = 0
            for row in reader:
                # Create/update Paper node
                s.run(UPSERT_PAPER, {
                    "doi": row["doi"],
                    "title": row["title"],
                    "abstract": row["abstract"],
                    "date": row["date"],  # YYYY-MM-DD
                })
                
                # Extract year from date (YYYY-MM-DD format)
                year = int(row["date"].split("-")[0])
                
                # Create Year node and PUBLISHED_IN relationship
                s.run(CREATE_YEAR_RELATIONSHIP, {
                    "doi": row["doi"],
                    "year": year
                })

                # LLM structured extraction with caching
                doi = row["doi"]
                title = row["title"]
                abstract = row["abstract"]
                cache_key = get_cache_key(doi, title, abstract)

                if cache_key in cache:
                    extraction = cache[cache_key]
                else:
                    result = extract_paper_entities(title, abstract, api_key=OPENAI_API_KEY, model=OPENAI_MODEL)
                    result.concepts = result.concepts[:MAX_CONCEPTS]
                    extraction = result.model_dump()
                    cache[cache_key] = extraction

                # Upsert Concept nodes and DISCUSSES
                for c in extraction.get("concepts", []):
                    name = c.get("name", "").strip()
                    importance = c.get("importance", "mentioned")
                    if not name:
                        continue
                    s.run(
                        """
                        MERGE (k:Concept {name: $name})
                        WITH k
                        MATCH (p:Paper {doi: $doi})
                        MERGE (p)-[r:DISCUSSES]->(k)
                        ON CREATE SET r.importance = $importance
                        """,
                        {"name": name, "doi": doi, "importance": importance}
                    )

                # Upsert Concept relationships as RELATES_TO with type property
                for rel in extraction.get("relationships", []):
                    a = rel.get("concept_a", "").strip()
                    b = rel.get("concept_b", "").strip()
                    rel_type = rel.get("relationship_type", "relates_to").lower()
                    if not a or not b or a == b:
                        continue
                    s.run(
                        """
                        MERGE (a:Concept {name: $a})
                        MERGE (b:Concept {name: $b})
                        MERGE (a)-[r:RELATES_TO {type: $type}]->(b)
                        """,
                        {"a": a, "b": b, "type": rel_type}
                    )

                paper_count += 1

        save_cache(CACHE_PATH, cache)

    print(f" Loaded {paper_count} papers into Neo4j.")
    print(" Created Year nodes, PUBLISHED_IN, DISCUSSES, and RELATES_TO relationships.")

if __name__ == "__main__":
    main()
