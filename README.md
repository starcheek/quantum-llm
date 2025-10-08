# Quantum LLM

A comprehensive toolkit for quantum research data collection, knowledge graph creation, and intelligent querying using Large Language Models.

## Features

- 🔬 **arXiv Data Collection**: Fetch research papers from arXiv with intelligent tier-based distribution
- 🕸️ **Knowledge Graph Creation**: Build Neo4j knowledge graphs from research papers
- 🤖 **Intelligent Querying**: Ask natural language questions about your research data
- 📊 **Status Monitoring**: Check system status and data health
- ⚙️ **Easy Setup**: One-command setup and configuration

## Installation

```bash
# Clone the repository
git clone https://github.com/quantum-research/quantum-llm.git
cd quantum-llm

# Install in development mode
pip install -e .

# Or install dependencies manually
pip install -r requirements.txt
```

## Quick Start

1. **Configure your settings** (edit `config.json`):
   ```json
   {
     "neo4j": {
       "uri": "bolt://localhost:7687",
       "user": "neo4j",
       "password": "your_password"
     },
     "llm": {
       "openai_api_key": "your_openai_api_key"
     }
   }
   ```

2. **Fetch research papers**:
   ```bash
   quantum-llm arxiv
   ```

3. **Create knowledge graph**:
   ```bash
   quantum-llm graph
   ```

4. **Query your data**:
   ```bash
   quantum-llm query "What are the latest developments in quantum computing?"
   ```

## Commands

### `quantum-llm arxiv`
Fetch research papers from arXiv API with intelligent tier-based distribution.

**Example:**
```bash
quantum-llm arxiv
```

### `quantum-llm graph`
Create knowledge graph from papers data.

**Example:**
```bash
quantum-llm graph
```

### `quantum-llm query`
Query the knowledge graph with natural language.

**Arguments:**
- `question`: The question to ask about your research data (optional - enters interactive mode if not provided)

**Options:**
- `--graph`: Force graph mode
- `--rag`: Force RAG mode  
- `--all-papers`: Use all papers for RAG queries

**Examples:**
```bash
# Interactive mode
quantum-llm query

# Direct question
quantum-llm query "What are the main applications of quantum machine learning?"

# Force graph mode
quantum-llm query --graph "Which year focused most on quantum computing?"

# Force RAG with all papers
quantum-llm query --rag --all-papers "Explain quantum machine learning advances"
```

## Configuration

The system uses `config.json` for configuration. Copy `config.example.json` to `config.json` and modify as needed:

```json
{
  "neo4j": {
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "your_password"
  },
  "openai": {
    "api_key": "your_openai_api_key"
  },
  "arxiv": {
    "target_count": 4000,
    "categories": ["quant-ph", "cs.ET", "cond-mat.mes-hall"]
  }
}
```

## Data Structure

The system creates the following data structure:

```
quantum-llm/
├── data/
│   ├── papers.csv          # Fetched research papers
│   └── extraction_cache.json # Entity extraction cache
├── quantum_llm/
│   ├── __init__.py
│   ├── cli.py              # Command line interface
│   ├── arxiv_data_fetcher.py
│   ├── create_knowledge_graph.py
│   └── quantum_query_system.py
├── config.json            # Configuration file
├── requirements.txt        # Dependencies
└── setup.py              # Package setup
```

## Requirements

- Python 3.7+
- Neo4j database
- OpenAI API key
- Internet connection for arXiv API

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- GitHub Issues: https://github.com/quantum-research/quantum-llm/issues
- Documentation: https://github.com/quantum-research/quantum-llm#readme