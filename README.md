# Northwind AI Workflow

## Graph Design
- **Orchestrator Agent**: Requirement parser → router → assembler → synthesizer with repair loops
- **RAG Agent**: TF-IDF document retrieval with graph distillation for relevant chunk filtering
- **SQL Agent**: LLM-driven schema pruning → SQL generation → execution with query normalization
- **Hybrid Flow**: RAG distillation → retrieval → SQL pruning → generation → execution → synthesis

## DSPy Optimization
Optimized **SQLGenerator** module using JSON-prompting approach:
- **Before**: DSPy structured outputs with parsing issues
- **After**: Direct JSON schema prompting with robust extraction
- **Delta**: Improved SQL generation reliability and reduced parsing errors

## Trade-offs & Assumptions
- **Schema Pruning**: LLM-based pruning may over-include tables to avoid missing joins
- **Repair Strategy**: Limited to 2 repair attempts to prevent infinite loops
- **TF-IDF Retrieval**: Simple cosine similarity without semantic embeddings for faster processing