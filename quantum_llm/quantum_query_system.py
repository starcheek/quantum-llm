"""
Quantum Research Knowledge Graph Query System

A unified system for querying quantum research papers using both graph-based
queries and traditional RAG approaches. Automatically detects the best method
for each query type.

Usage:
    python3 quantum_query_system.py                    # Interactive mode
    python3 quantum_query_system.py "your question"   # Command line
    python3 quantum_query_system.py "question" --graph # Force graph mode
    python3 quantum_query_system.py "question" --rag   # Force RAG mode
"""

import json
import re
import sys
from typing import List, Dict, Tuple, Any, Optional

from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from openai import OpenAI


class Config:
    """Configuration management."""
    
    def __init__(self, config_path: str = 'config.json'):
        with open(config_path, 'r') as f:
            self.data = json.load(f)
    
    @property
    def neo4j_uri(self) -> str:
        return self.data["neo4j"]["uri"]
    
    @property
    def neo4j_user(self) -> str:
        return self.data["neo4j"]["user"]
    
    @property
    def neo4j_password(self) -> str:
        return self.data["neo4j"]["password"]
    
    @property
    def neo4j_database(self) -> str:
        return self.data["neo4j"]["database"]
    
    @property
    def openai_api_key(self) -> str:
        return self.data["llm"]["openai_api_key"]
    
    @property
    def openai_model(self) -> str:
        return self.data["llm"]["openai_model"]


class Neo4jClient:
    """Neo4j database client."""
    
    def __init__(self, config: Config):
        self.config = config
        self.driver = GraphDatabase.driver(
            config.neo4j_uri, 
            auth=(config.neo4j_user, config.neo4j_password)
        )
    
    def get_session(self):
        return self.driver.session(database=self.config.neo4j_database)
    
    def close(self):
        self.driver.close()
    
    def run_query(self, cypher: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Execute a Cypher query and return results."""
        with self.get_session() as session:
            try:
                records = list(session.run(cypher, parameters or {}))
                return [record.data() for record in records]
            except Exception as e:
                return [{"error": str(e), "cypher": cypher}]
    
    def get_concepts(self) -> List[str]:
        """Get all available concepts from the database."""
        result = self.run_query("MATCH (c:Concept) RETURN c.name as name ORDER BY name")
        return [record["name"] for record in result]


class GraphQueryEngine:
    """Graph-based query engine using Neo4j Cypher."""
    
    def __init__(self, neo4j_client: Neo4jClient, config: Config):
        self.neo4j = neo4j_client
        self.config = config
        self.openai_client = OpenAI(api_key=config.openai_api_key)
    
    def _create_schema_hint(self, concepts: List[str]) -> str:
        """Create enhanced schema hint with actual concept names."""
        concepts_list = "\n".join([f"  - {concept}" for concept in concepts])
        
        return f"""
You translate user questions to Cypher for Neo4j. Use ONLY this schema:

Nodes:
  - Paper {{arxiv_id, title, abstract, date}}
  - Year {{year}}
  - Concept {{name}}

Relationships:
  - (Paper)-[:PUBLISHED_IN]->(Year)
  - (Paper)-[:DISCUSSES {{importance}}] ->(Concept)
  - (Concept)-[:RELATES_TO {{type}}] ->(Concept)

Available Concepts (use EXACT names):
{concepts_list}

CRITICAL RULES:
1. ALWAYS use the EXACT concept names from the list above (case-sensitive).
2. For concept matching, use: MATCH (p:Paper)-[:DISCUSSES]->(c:Concept {{name: "Exact Concept Name"}})
3. For time questions, use Year nodes: MATCH (p:Paper)-[:PUBLISHED_IN]->(y:Year)
4. For relationships between concepts, use: MATCH (c1:Concept {{name: "Concept1"}})-[:RELATES_TO]->(c2:Concept {{name: "Concept2"}})
5. For co-occurrence, find papers discussing both: MATCH (p:Paper)-[:DISCUSSES]->(c1:Concept {{name: "Concept1"}}), (p)-[:DISCUSSES]->(c2:Concept {{name: "Concept2"}})
6. Always use proper MATCH clauses - don't use WHERE with pattern expressions.
7. Limit outputs reasonably (e.g., LIMIT 25) for readability.
8. Always return clear, tabular fields.

EXAMPLES:
- "Which year focused on X?" → MATCH (p:Paper)-[:PUBLISHED_IN]->(y:Year), (p)-[:DISCUSSES]->(c:Concept {{name: "X"}}) RETURN y.year, COUNT(p) ORDER BY COUNT(p) DESC LIMIT 1
- "Papers about X in year Y" → MATCH (p:Paper)-[:PUBLISHED_IN]->(y:Year {{year: Y}}), (p)-[:DISCUSSES]->(c:Concept {{name: "X"}}) RETURN p.title, p.arxiv_id, p.date
- "Concepts related to X" → MATCH (c:Concept {{name: "X"}})-[:RELATES_TO]->(related:Concept) RETURN related.name
"""
    
    def _generate_cypher(self, question: str, concepts: List[str]) -> Tuple[str, str]:
        """Generate Cypher query with validation."""
        schema_hint = self._create_schema_hint(concepts)
        
        messages = [
            {"role": "system", "content": schema_hint},
            {
                "role": "user",
                "content": (
                    "Translate the question to a single Cypher query. "
                    "Use EXACT concept names from the provided list. "
                    "Follow the examples and rules exactly. "
                    "Return ONLY the Cypher, no explanations.\n\nQuestion: "
                    + question
                ),
            },
        ]
        
        try:
            resp = self.openai_client.chat.completions.create(
                model=self.config.openai_model,
                temperature=0.0,
                messages=messages,
            )
            cypher = resp.choices[0].message.content.strip()
            
            # Clean up response
            if cypher.startswith("```"):
                cypher = cypher.strip('`')
                parts = cypher.split('\n', 1)
                cypher = parts[1] if len(parts) > 1 else cypher
            
            cypher = cypher.strip()
            
            # Validate concept names
            concept_names = re.findall(r'Concept \{name: "([^"]+)"\}', cypher)
            invalid_concepts = [name for name in concept_names if name not in concepts]
            
            if invalid_concepts:
                return self._generate_fallback_cypher(question, concepts)
            
            return cypher, "Generated with enhanced schema"
            
        except Exception:
            return self._generate_fallback_cypher(question, concepts)
    
    def _generate_fallback_cypher(self, question: str, concepts: List[str]) -> Tuple[str, str]:
        """Generate fallback Cypher query based on question patterns."""
        question_lower = question.lower()
        mentioned_concepts = [c for c in concepts if c.lower() in question_lower]
        
        if "year" in question_lower and "most" in question_lower and mentioned_concepts:
            concept_name = mentioned_concepts[0]
            return f'''
MATCH (p:Paper)-[:PUBLISHED_IN]->(y:Year), (p)-[:DISCUSSES]->(c:Concept {{name: "{concept_name}"}})
RETURN y.year, COUNT(p) AS paper_count
ORDER BY paper_count DESC
LIMIT 1
''', f"Fallback: year analysis for {concept_name}"
        
        elif "related" in question_lower and mentioned_concepts:
            concept_name = mentioned_concepts[0]
            return f'''
MATCH (c:Concept {{name: "{concept_name}"}})-[:RELATES_TO]->(related:Concept)
RETURN related.name AS related_concept
LIMIT 25
''', f"Fallback: relationships for {concept_name}"
        
        elif "co-occur" in question_lower and mentioned_concepts:
            concept_name = mentioned_concepts[0]
            return f'''
MATCH (p:Paper)-[:DISCUSSES]->(c1:Concept {{name: "{concept_name}"}}), (p)-[:DISCUSSES]->(c2:Concept)
WHERE c2.name <> "{concept_name}"
RETURN c2.name AS co_occurring_concept, COUNT(*) AS frequency
ORDER BY frequency DESC
LIMIT 25
''', f"Fallback: co-occurrence for {concept_name}"
        
        # Default fallback
        return '''
MATCH (p:Paper)-[:DISCUSSES]->(c:Concept)
RETURN c.name AS concept, COUNT(p) AS paper_count
ORDER BY paper_count DESC
LIMIT 10
''', "Default fallback: top concepts"
    
    def _format_results(self, results: List[Dict]) -> str:
        """Format query results for display."""
        if not results:
            return "No results found."
        
        if "error" in results[0]:
            return f"Error: {results[0]['error']}"
        
        if len(results) == 1 and len(results[0]) == 1:
            key, value = list(results[0].items())[0]
            return f"{key}: {value}"
        
        formatted_lines = []
        for i, row in enumerate(results[:10], 1):
            if len(row) == 1:
                key, value = list(row.items())[0]
                formatted_lines.append(f"{i}. {key}: {value}")
            else:
                parts = [f"{key}: {value}" for key, value in row.items()]
                formatted_lines.append(f"{i}. {' | '.join(parts)}")
        
        if len(results) > 10:
            formatted_lines.append(f"... and {len(results) - 10} more results")
        
        return "\n".join(formatted_lines)
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process a natural language query using graph approach."""
        concepts = self.neo4j.get_concepts()
        cypher, method = self._generate_cypher(question, concepts)
        results = self.neo4j.run_query(cypher)
        formatted_results = self._format_results(results)
        
        return {
            "question": question,
            "cypher": cypher,
            "method": method,
            "results": results,
            "formatted_results": formatted_results,
            "status": f"Query executed successfully, returned {len(results)} results" if results else "No results found"
        }


class RAGEngine:
    """Traditional RAG engine using keyword search and LLM synthesis."""
    
    def __init__(self, neo4j_client: Neo4jClient, config: Config):
        self.neo4j = neo4j_client
        self.config = config
        self.llm = ChatOpenAI(
            model=config.openai_model,
            api_key=config.openai_api_key,
            temperature=0.3
        )
        self.answer_llm = ChatOpenAI(
            model=config.openai_model,
            api_key=config.openai_api_key,
            temperature=0.1
        )
    
    def _generate_search_queries(self, question: str) -> List[str]:
        """Generate search keywords from question."""
        prompt = PromptTemplate.from_template("""
You are a research assistant helping to find relevant academic papers. 
Given a research question, generate 3-5 different search terms that would help find relevant papers.

Each search term should be:
- A key concept or technical term from the question
- Different aspects or synonyms of the main topic
- Specific enough to find relevant papers

Question: {question}

Generate search terms (one per line, no numbering):
""")
        
        chain = prompt | self.llm
        result = chain.invoke({"question": question})
        
        queries = [q.strip() for q in result.content.strip().split('\n') if q.strip()]
        return queries[:5]
    
    def _search_papers(self, search_queries: List[str]) -> List[Dict]:
        """Search papers using keywords."""
        search_conditions = []
        for query in search_queries:
            condition = f"(toLower(p.title) CONTAINS toLower('{query}') OR toLower(p.abstract) CONTAINS toLower('{query}'))"
            search_conditions.append(condition)
        
        where_clause = " OR ".join(search_conditions)
        cypher = f"""
MATCH (p:Paper)
WHERE {where_clause}
RETURN p.arxiv_id as arxiv_id, p.title as title, p.abstract as abstract, toString(p.date) as date
ORDER BY p.date DESC
LIMIT 50
"""
        
        return self.neo4j.run_query(cypher)
    
    def _get_all_papers(self) -> List[Dict]:
        """Get all papers from database."""
        cypher = """
MATCH (p:Paper)
RETURN p.arxiv_id as arxiv_id, p.title as title, p.abstract as abstract, toString(p.date) as date
ORDER BY p.date DESC
LIMIT 100
"""
        return self.neo4j.run_query(cypher)
    
    def _synthesize_answer(self, question: str, papers: List[Dict]) -> str:
        """Synthesize answer from papers using LLM."""
        if not papers:
            return "No relevant papers found."
        
        context_parts = []
        for paper in papers[:10]:
            paper_text = f"Title: {paper['title']}\nDate: {paper['date']}\nAbstract: {paper['abstract']}"
            context_parts.append(paper_text)
        
        context = "\n\n".join(context_parts)
        
        template = """You are a concise research assistant.
Use the provided context (paper titles/abstracts) to answer the user question.
Cite the arXiv IDs of the most relevant papers at the end under "Sources".
If unsure, say you are unsure.

Question:
{question}

Context:
{context}
"""
        
        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.answer_llm
        result = chain.invoke({"question": question, "context": context})
        
        return result.content.strip()
    
    def query(self, question: str, use_all_papers: bool = False) -> Dict[str, Any]:
        """Process a natural language query using RAG approach."""
        if use_all_papers:
            papers = self._get_all_papers()
            method = "All papers search"
        else:
            search_queries = self._generate_search_queries(question)
            papers = self._search_papers(search_queries)
            method = f"Keyword search: {search_queries}"
        
        answer = self._synthesize_answer(question, papers)
        sources = [paper['arxiv_id'] for paper in papers[:5]]
        
        return {
            "question": question,
            "method": method,
            "papers_found": len(papers),
            "answer": answer,
            "sources": sources,
            "status": f"Found {len(papers)} papers" if papers else "No papers found"
        }


class QueryRouter:
    """Routes queries to the appropriate engine."""
    
    @staticmethod
    def detect_query_type(question: str) -> str:
        """Detect whether to use graph queries or RAG."""
        question_lower = question.lower()
        
        graph_keywords = [
            "which year", "when", "how many", "count", "relationship", "related to",
            "co-occur", "path", "between", "top", "most", "frequently", "trend",
            "per year", "by year", "concepts", "papers about", "papers discussing"
        ]
        
        rag_keywords = [
            "what are", "how do", "explain", "describe", "tell me about", "latest advances",
            "improve", "benefits", "advantages", "disadvantages", "compare", "difference"
        ]
        
        graph_score = sum(1 for keyword in graph_keywords if keyword in question_lower)
        rag_score = sum(1 for keyword in rag_keywords if keyword in question_lower)
        
        if graph_score > rag_score:
            return "graph"
        elif rag_score > graph_score:
            return "rag"
        else:
            return "rag" if any(word in question_lower for word in ["?", "what", "how", "why", "explain"]) else "graph"


class QuantumQuerySystem:
    """Main query system that integrates both approaches."""
    
    def __init__(self, config_path: str = 'config.json'):
        self.config = Config(config_path)
        self.neo4j = Neo4jClient(self.config)
        self.graph_engine = GraphQueryEngine(self.neo4j, self.config)
        self.rag_engine = RAGEngine(self.neo4j, self.config)
        self.router = QueryRouter()
    
    def query(self, question: str, force_mode: Optional[str] = None, use_all_papers: bool = False) -> Dict[str, Any]:
        """Process a query using the best available approach."""
        query_type = force_mode or self.router.detect_query_type(question)
        
        if query_type == "graph":
            return self.graph_engine.query(question)
        else:
            return self.rag_engine.query(question, use_all_papers)
    
    def close(self):
        """Close database connections."""
        self.neo4j.close()


def print_help():
    """Print help information."""
    print(" Help - Query Types:")
    print("\nGraph Queries (structured data):")
    print("  - Which year focused most on [concept]?")
    print("  - What are the top discussed concepts?")
    print("  - Show concepts related to [concept]")
    print("  - Which concepts co-occur with [concept]?")
    print("\nRAG Queries (explanations):")
    print("  - What are the latest advances in [topic]?")
    print("  - How do [concept1] improve [concept2]?")
    print("  - Explain the benefits of [concept]")
    print("  - Compare [concept1] and [concept2]")
    print("\nThe system automatically detects the best approach!")


def interactive_mode():
    """Run interactive query mode."""
    print("Quantum Research Query System")
    print("Ask questions about quantum research papers!")
    print("\nThe system automatically chooses the best approach:")
    print("  - Graph Queries: For structured data (years, counts, relationships)")
    print("  - RAG Queries: For explanations and analysis")
    print("\nType 'help' for more info, 'quit' to exit.")
    
    system = QuantumQuerySystem()
    
    try:
        while True:
            question = input("\nYour question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if question.lower() == 'help':
                print_help()
                continue
            
            if question.lower() == 'concepts':
                concepts = system.neo4j.get_concepts()
                print(f"\nAvailable Concepts ({len(concepts)} total):")
                for i, concept in enumerate(concepts, 1):
                    print(f"  {i:2d}. {concept}")
                print()
                continue
            
            # Process the query
            result = system.query(question)
            
            print(f"Question: {question}")
            
            if "cypher" in result:
                # Graph query result
                print(f"Generated Cypher ({result['method']}):")
                print(result['cypher'])
                print(f"Status: {result['status']}")
                print("Results:")
                print(result['formatted_results'])
            else:
                # RAG query result
                print(f"Method: {result['method']}")
                print(f"Status: {result['status']}")
                print("Answer:")
                print(result['answer'])
                print("Sources:")
                for source in result['sources']:
                    print(f"  - {source}")
    
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        system.close()


def main(question: Optional[str] = None, graph: bool = False, rag: bool = False, all_papers: bool = False):
    """Main entry point. Arguments are passed directly from the CLI.

    - If no question is provided, start interactive mode.
    - graph/rag flags choose the engine when provided; graph > rag if both set.
    - all_papers toggles RAG to use the full papers set.
    """
    if not question:
        interactive_mode()
        return

    force_mode = "graph" if graph else ("rag" if rag else None)

    system = QuantumQuerySystem()
    try:
        result = system.query(question.strip(), force_mode, all_papers)

        print(f"Question: {question.strip()}")

        if "cypher" in result:
            print(f"Generated Cypher ({result['method']}):")
            print(result['cypher'])
            print(f"Status: {result['status']}")
            print("Results:")
            print(result['formatted_results'])
        else:
            print(f"Method: {result['method']}")
            print(f"Status: {result['status']}")
            print("Answer:")
            print(result['answer'])
            print("Sources:")
            for source in result['sources']:
                print(f"  - {source}")
    finally:
        system.close()


if __name__ == "__main__":
    main()
