#!/usr/bin/env python3
"""
Quantum LLM CLI - Minimal runners for project scripts
"""

import typer
import sys
import os

# Add the current directory to Python path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arxiv_data_fetcher import main as fetch_main
from create_knowledge_graph import main as graph_main
from quantum_query_system import main as query_main

app = typer.Typer(add_completion=False)

@app.command()
def arxiv():
    fetch_main()

@app.command()
def graph():
    graph_main()

@app.command()
def query(
    question: str = typer.Argument(None),
    graph: bool = typer.Option(False, "--graph", help="Force graph mode"),
    rag: bool = typer.Option(False, "--rag", help="Force RAG mode"),
    all_papers: bool = typer.Option(False, "--all-papers", help="Use all papers")
):
    query_main(question=question, graph=graph, rag=rag, all_papers=all_papers)

if __name__ == "__main__":
    app()
