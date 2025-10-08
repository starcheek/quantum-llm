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

1. **Setup the system**:
   ```bash
   quantum-llm setup
   ```

2. **Configure your settings** (edit `config.json`):
   ```json
   {
     "neo4j": {
       "uri": "bolt://localhost:7687",
       "user": "neo4j",
       "password": "your_password"
     },
     "openai": {
       "api_key": "your_openai_api_key"
     }
   }
   ```

3. **Fetch research papers**:
   ```bash
   quantum-llm fetch --target 4000
   ```

4. **Create knowledge graph**:
   ```bash
   quantum-llm graph
   ```

5. **Query your data**:
   ```bash
   quantum-llm query "What are the latest developments in quantum computing?"
   ```

## Commands

### `quantum-llm fetch`
Fetch research papers from arXiv API.

**Options:**
- `--target`: Number of papers to fetch (default: 4000)
- `--clean`: Clean existing data before fetching (default: true)
- `--verbose`: Enable verbose output

**Example:**
```bash
quantum-llm fetch --target 5000 --verbose
```

### `quantum-llm graph`
Create knowledge graph from papers data.

**Options:**
- `--config`: Path to configuration file (default: config.json)
- `--verbose`: Enable verbose output

**Example:**
```bash
quantum-llm graph --config my_config.json
```

### `quantum-llm query`
Query the knowledge graph with natural language.

**Arguments:**
- `question`: The question to ask about your research data

**Options:**
- `--config`: Path to configuration file (default: config.json)
- `--verbose`: Enable verbose output

**Example:**
```bash
quantum-llm query "What are the main applications of quantum machine learning?"
```

### `quantum-llm status`
Check the status of your quantum-llm system.

**Example:**
```bash
quantum-llm status
```

### `quantum-llm setup`
Initial setup for the quantum-llm system.

**Example:**
```bash
quantum-llm setup
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