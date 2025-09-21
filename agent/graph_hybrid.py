"""LangGraph hybrid agent with orchestrator, RAG, SQL, and synthesizer agents."""

import json
import logging
import datetime
from typing import Dict, Any, List, Optional, Annotated, TypedDict
from pathlib import Path
import operator as op

from langgraph.graph import StateGraph, END
from pydantic import BaseModel

from .dspy_signatures import (
    RequirementParser, Router, SQLGenerator, Synthesizer,
    RequirementParserOutput, RouterOutput, SQLGeneratorOutput, SynthesizerOutput
)
from .rag.retrieval import RAGRetriever, RAGGraphDistillerNode, RAGRetrieverNode
from .sql.sqlite_tool import (
    SQLiteInspector, SQLExecutor, SQLGraphPrunerNode, 
    SQLGeneratorNode, SQLExecutorNode
)


def _ensure_error_logs_dir() -> Path:
    """Ensure error_logs directory exists at project root."""
    try:
        base_dir = Path(__file__).resolve().parents[1]
        logs_dir = base_dir / 'error_logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir
    except Exception:
        # Last-resort: current file directory
        return Path(__file__).resolve().parent


def _write_error_log(component: str, error: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Write a concise JSON error log file."""
    try:
        logs_dir = _ensure_error_logs_dir()
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = logs_dir / f"{component}_{ts}.json"
        payload = {
            "timestamp": ts,
            "component": component,
            "error": str(error),
        }
        if details:
            # Keep details concise
            payload["details"] = details
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False))
    except Exception:
        # Never raise from logger
        pass


class AgentState(TypedDict, total=False):
    """State for the hybrid agent (TypedDict with reducers for LangGraph 0.2)."""
    # Input
    question: str
    format_hint: str
    id: str
    
    # Requirement Parser outputs
    ontological_blocks: List[Dict[str, Any]]
    expected_answer_type: str
    date_ranges: List[str]
    kpi_formulas: List[str]
    categories_entities: List[Dict[str, Any]]
    
    # Router outputs
    approach: str
    router_confidence: float
    router_reasoning: str
    
    # RAG outputs
    rag_graph_repr: Dict[str, Any]
    rag_search_tags: List[str]
    rag_chunks: List[Dict[str, Any]]
    rag_chunk_count: int
    rag_top_k: int
    
    # SQL outputs
    pruned_schema: str
    generated_sql: str
    sql_explanation: str
    expected_columns: List[str]
    sql_results: List[Dict[str, Any]]
    sql_success: bool
    sql_error: Optional[str]
    sql_row_count: int
    sql_columns: List[str]
    previous_sql: Optional[str]
    
    # Assembler outputs
    assembled_results: Annotated[Dict[str, Any], op.or_]
    
    # Synthesizer outputs
    final_answer: Any
    confidence: float
    explanation: str
    citations: List[str]
    
    # Repair tracking
    repair_count: int
    max_repairs: int
    repair_type: str  # "sql_error", "invalid_output", "invalid_citations"
    previous_attempt: Optional[str]
    error_message: Optional[str]
    
    # Logging
    trace_log: Annotated[List[Dict[str, Any]], op.add]


class RequirementParserNode:
    """Orchestrator Agent: Requirement Parser Node."""
    
    def __init__(self, requirement_parser: RequirementParser):
        self.requirement_parser = requirement_parser
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Parse requirements from natural language question."""
        question = state.get("question", "")
        format_hint = state.get("format_hint", "")
        print("Node start: RequirementParser")
        
        # Log the operation
        trace_entry = {
            "node": "RequirementParser",
            "input": {"question": question, "format_hint": format_hint},
            "timestamp": str(datetime.datetime.now())
        }
        
        try:
            result = self.requirement_parser.forward(question, format_hint)
        except Exception as e:
            trace_entry["error"] = str(e)
            trace_entry["success"] = False
            try:
                _write_error_log(
                    component="RequirementParserNode",
                    error=str(e),
                    details={
                        "question_id": state.get("id", ""),
                        "question": (question[:200] + '...') if len(question) > 200 else question,
                        "format_hint": format_hint,
                    }
                )
            except Exception:
                pass
            return {
                **state,
                "trace_log": state.get("trace_log", []) + [trace_entry],
                "error_message": str(e)
            }
        
        # Build new state from successful result
        new_state = {
            **state,
            "ontological_blocks": result.ontological_blocks,
            "expected_answer_type": result.expected_answer_type,
            "date_ranges": result.date_ranges,
            "kpi_formulas": result.kpi_formulas,
            "categories_entities": result.categories_entities,
        }
        
        # Best-effort trace logging
        try:
            trace_entry["output"] = result.model_dump()
            trace_entry["success"] = True
            new_state["trace_log"] = state.get("trace_log", []) + [trace_entry]
        except Exception:
            pass
        
        print("Node end: RequirementParser")
        return new_state


class RouterNode:
    """Orchestrator Agent: Router Node."""
    
    def __init__(self, router: Router):
        self.router = router
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Route question to appropriate approach."""
        question = state.get("question", "")
        ontological_blocks = state.get("ontological_blocks", [])
        expected_answer_type = state.get("expected_answer_type", "")
        print("Node start: Router")
        
        trace_entry = {
            "node": "Router",
            "input": {
                "question": question,
                "ontological_blocks": ontological_blocks,
                "expected_answer_type": expected_answer_type
            },
            "timestamp": str(datetime.datetime.now())
        }
        
        try:
            result = self.router.forward(question, ontological_blocks, expected_answer_type)
        except Exception as e:
            trace_entry["error"] = str(e)
            trace_entry["success"] = False
            try:
                _write_error_log(
                    component="RouterNode",
                    error=str(e),
                    details={
                        "question_id": state.get("id", ""),
                        "expected_answer_type": expected_answer_type,
                        "ontological_blocks_len": len(ontological_blocks),
                    }
                )
            except Exception:
                pass
            result_state = {
                **state,
                "approach": "hybrid",  # Default fallback
                "router_confidence": 0.5,
                "router_reasoning": f"Fallback due to error: {str(e)}",
            }
            result_state["trace_log"] = state.get("trace_log", []) + [trace_entry]
            print(f"Node end: Router (fallback) -> approach={result_state.get('approach')}")
            return result_state
        
        result_state = {
            **state,
            "approach": result.approach,
            "router_confidence": result.confidence,
            "router_reasoning": result.reasoning,
        }
        try:
            trace_entry["output"] = result.model_dump()
            trace_entry["success"] = True
            result_state["trace_log"] = state.get("trace_log", []) + [trace_entry]
        except Exception:
            pass
        print(f"Node end: Router -> approach={result_state.get('approach')}")
        return result_state


class AssemblerNode:
    """Orchestrator Agent: Assembler Node."""
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble outputs from RAG and/or SQL agents."""
        approach = state.get("approach", "hybrid")
        print("Node start: Assembler")
        
        assembled_results = {}
        
        if approach in ["rag", "hybrid"]:
            assembled_results["rag_data"] = {
                "chunks": list({c["id"]: c for c in state.get("rag_chunks", [])}.values()),
                "chunk_count": len(state.get("rag_chunks", [])),
                "search_tags": list(dict.fromkeys(state.get("rag_search_tags", [])))
            }
        
        if approach in ["sql", "hybrid"]:
            assembled_results["sql_data"] = {
                "results": state.get("sql_results", []),
                "success": state.get("sql_success", False),
                "error": state.get("sql_error"),
                "query": state.get("generated_sql", ""),
                "row_count": state.get("sql_row_count", 0)
            }
        
        trace_entry = {
            "node": "Assembler",
            "input": {"approach": approach},
            "output": assembled_results,
            "success": True,
            "timestamp": str(datetime.datetime.now())
        }
        
        # Log SQL execution errors concisely (surfaced from SQLExecutorNode)
        try:
            if approach in ["sql", "hybrid"] and state.get("sql_error"):
                _write_error_log(
                    component="SQLExecutorNode",
                    error=str(state.get("sql_error")),
                    details={
                        "question_id": state.get("id", ""),
                        "query": state.get("generated_sql", "")[:300],
                        "row_count": state.get("sql_row_count", 0),
                    }
                )
        except Exception:
            pass

        print("Node end: Assembler")
        return {
            **state,
            "assembled_results": assembled_results,
            "trace_log": state.get("trace_log", []) + [trace_entry]
        }


class SynthesizerNode:
    """Answer Synthesizer Agent: Synthesizer Node."""
    
    def __init__(self, synthesizer: Synthesizer):
        self.synthesizer = synthesizer
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize final answer from assembled results."""
        question = state.get("question", "")
        format_hint = state.get("format_hint", "")
        assembled_results = state.get("assembled_results", {})
        ontological_blocks = state.get("ontological_blocks", [])
        print("Node start: Synthesizer")
        
        # Extract SQL results and RAG chunks
        sql_results = assembled_results.get("sql_data", {}).get("results", [])
        rag_chunks = assembled_results.get("rag_data", {}).get("chunks", [])
        
        # Debug logging for empty sql_results
        if not sql_results:
            sql_success = assembled_results.get("sql_data", {}).get("success", False)
            sql_error = assembled_results.get("sql_data", {}).get("error", "")
            print(f"WARNING: Empty sql_results. SQL success: {sql_success}, SQL error: {sql_error}")
            if not sql_success and sql_error:
                print(f"SQL execution failed: {sql_error}")
        else:
            print(f"Found {len(sql_results)} SQL results")
        
        # Check for repair context
        previous_attempt = state.get("previous_attempt")
        error_message = state.get("error_message")
        
        trace_entry = {
            "node": "Synthesizer",
            "input": {
                "question": question,
                "format_hint": format_hint,
                "sql_results_count": len(sql_results),
                "rag_chunks_count": len(rag_chunks),
                "is_repair": bool(previous_attempt)
            },
            "timestamp": str(datetime.datetime.now())
        }
        
        try:
            result = self.synthesizer.forward(
                question=question,
                format_hint=format_hint,
                sql_results=sql_results,
                rag_chunks=rag_chunks,
                ontological_blocks=ontological_blocks,
                previous_attempt=previous_attempt,
                error_message=error_message
            )
            
            trace_entry["output"] = {
                "final_answer": str(result.final_answer)[:200],  # Truncate for logging
                "confidence": result.confidence,
                "explanation": result.explanation,
                "citations": result.citations
            }
            trace_entry["success"] = True
            
            result_state = {
                **state,
                "final_answer": result.final_answer,
                "confidence": result.confidence,
                "explanation": result.explanation,
                "citations": result.citations,
                "trace_log": state.get("trace_log", []) + [trace_entry]
            }
            print("Node end: Synthesizer")
            return result_state
            
        except Exception as e:
            trace_entry["error"] = str(e)
            trace_entry["success"] = False
            try:
                _write_error_log(
                    component="SynthesizerNode",
                    error=str(e),
                    details={
                        "question_id": state.get("id", ""),
                        "sql_results_count": len(sql_results),
                        "rag_chunks_count": len(rag_chunks),
                    }
                )
            except Exception:
                pass
            
            result_state = {
                **state,
                "final_answer": None,
                "confidence": 0.0,
                "explanation": f"Synthesis failed: {str(e)}",
                "citations": [],
                "error_message": str(e),
                "trace_log": state.get("trace_log", []) + [trace_entry]
            }
            print("Node end: Synthesizer (error)")
            return result_state


class RepairNode:
    """Repair node for handling errors and invalid outputs."""
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Determine repair strategy and update state for retry."""
        repair_count = state.get("repair_count", 0)
        max_repairs = state.get("max_repairs", 2)
        print("Node start: Repair")
        
        if repair_count >= max_repairs:
            # Max repairs reached, return current state
            return {
                **state,
                "repair_type": "max_reached"
            }
        
        # Determine repair type
        sql_error = state.get("sql_error")
        final_answer = state.get("final_answer")
        format_hint = state.get("format_hint", "")
        
        repair_type = ""
        error_message = ""
        
        if sql_error:
            repair_type = "sql_error"
            error_message = sql_error
        elif final_answer is None:
            repair_type = "invalid_output"
            error_message = "Failed to generate valid output"
        elif not self._validate_format(final_answer, format_hint):
            repair_type = "invalid_format"
            error_message = f"Output format doesn't match {format_hint}"
        elif not state.get("citations"):
            repair_type = "invalid_citations"
            error_message = "Missing or invalid citations"
        
        trace_entry = {
            "node": "Repair",
            "repair_type": repair_type,
            "repair_count": repair_count + 1,
            "error_message": error_message,
            "timestamp": str(datetime.datetime.now())
        }
        
        result_state = {
            **state,
            "repair_count": repair_count + 1,
            "repair_type": repair_type,
            "error_message": error_message,
            "previous_attempt": str(final_answer) if final_answer else "",
            "previous_sql": state.get("generated_sql"),
            "trace_log": state.get("trace_log", []) + [trace_entry]
        }
        print(f"Node end: Repair -> type={repair_type}")
        return result_state
    
    def _validate_format(self, answer: Any, format_hint: str) -> bool:
        """Validate if answer matches format hint."""
        if not format_hint:
            return True
        
        try:
            if format_hint == "int":
                return isinstance(answer, int)
            elif format_hint == "float":
                return isinstance(answer, (int, float))
            elif format_hint.startswith("list[") or format_hint.startswith("List["):
                return isinstance(answer, list)
            elif format_hint.startswith("{") or format_hint == "dict":
                return isinstance(answer, dict)
            else:
                return True  # Unknown format, assume valid
        except:
            return False


class HybridAgent:
    """Main hybrid agent orchestrating all components."""
    
    def __init__(self, db_path: str, docs_dir: str, model_name: str = "phi3.5:3.8b-mini-instruct-q4_K_M"):
        self.db_path = Path(db_path)
        self.docs_dir = Path(docs_dir)
        self.model_name = model_name
        
        # Initialize DSPy
        import dspy
        self.lm = dspy.LM(f"ollama/{model_name}", max_tokens=1000, temperature=0.1)
        dspy.settings.configure(lm=self.lm)
        
        # Initialize components
        self.requirement_parser = RequirementParser()
        self.router = Router()
        self.sql_generator = SQLGenerator()
        self.synthesizer = Synthesizer()
        
        # Initialize tools
        self.rag_retriever = RAGRetriever(str(docs_dir))
        self.sql_inspector = SQLiteInspector(str(db_path))
        self.sql_executor = SQLExecutor(str(db_path))
        
        # Initialize nodes
        self.requirement_parser_node = RequirementParserNode(self.requirement_parser)
        self.router_node = RouterNode(self.router)
        self.rag_distiller_node = RAGGraphDistillerNode(self.rag_retriever)
        self.rag_retriever_node = RAGRetrieverNode(self.rag_retriever)
        self.sql_pruner_node = SQLGraphPrunerNode(self.sql_inspector)
        self.sql_generator_node = SQLGeneratorNode(self.sql_generator)
        self.sql_executor_node = SQLExecutorNode(self.sql_executor)
        self.assembler_node = AssemblerNode()
        self.synthesizer_node = SynthesizerNode(self.synthesizer)
        self.repair_node = RepairNode()
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Initialize StateGraph with the TypedDict state schema
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("requirement_parser", self.requirement_parser_node)
        graph.add_node("router", self.router_node)
        graph.add_node("rag_distiller", self.rag_distiller_node)
        graph.add_node("rag_retriever", self.rag_retriever_node)
        graph.add_node("sql_pruner", self.sql_pruner_node)
        graph.add_node("sql_generator", self.sql_generator_node)
        graph.add_node("sql_executor", self.sql_executor_node)
        graph.add_node("assembler", self.assembler_node)
        graph.add_node("synthesizer", self.synthesizer_node)
        graph.add_node("repair", self.repair_node)
        
        # Set entry point
        graph.set_entry_point("requirement_parser")
        
        # Add edges
        graph.add_edge("requirement_parser", "router")
        
        # Conditional routing after router
        def route_after_router(state: Dict[str, Any]) -> str:
            approach = state.get("approach", "hybrid")
            if approach == "rag":
                return "rag_distiller"
            elif approach == "sql":
                return "sql_pruner"
            else:  # hybrid
                return "rag_distiller"  # Start with RAG for hybrid
        
        graph.add_conditional_edges(
            "router",
            route_after_router,
            {
                "rag_distiller": "rag_distiller",
                "sql_pruner": "sql_pruner"
            }
        )
        
        # RAG flow
        graph.add_edge("rag_distiller", "rag_retriever")
        
        # SQL flow
        graph.add_edge("sql_pruner", "sql_generator")
        graph.add_edge("sql_generator", "sql_executor")
        
        # Conditional routing after RAG retriever
        def route_after_rag(state: Dict[str, Any]) -> str:
            approach = state.get("approach", "hybrid")
            if approach == "hybrid":
                return "sql_pruner"
            else:
                return "assembler"
        
        graph.add_conditional_edges(
            "rag_retriever",
            route_after_rag,
            {
                "sql_pruner": "sql_pruner",
                "assembler": "assembler"
            }
        )
        
        # Route to assembler after SQL executor
        graph.add_edge("sql_executor", "assembler")
        
        # Assembler to synthesizer
        graph.add_edge("assembler", "synthesizer")
        
        # Conditional routing after synthesizer (repair logic)
        def route_after_synthesizer(state: Dict[str, Any]) -> str:
            sql_error = state.get("sql_error")
            final_answer = state.get("final_answer")
            repair_count = state.get("repair_count", 0)
            max_repairs = state.get("max_repairs", 2)
            
            # Check if repair is needed
            if repair_count >= max_repairs:
                return END
            
            if sql_error or final_answer is None:
                return "repair"
            
            # Validate format and citations
            format_hint = state.get("format_hint", "")
            citations = state.get("citations", [])
            
            if not citations or not self._validate_output_format(final_answer, format_hint):
                return "repair"
            
            return END
        
        graph.add_conditional_edges(
            "synthesizer",
            route_after_synthesizer,
            {
                "repair": "repair",
                END: END
            }
        )
        
        # Repair routing
        def route_after_repair(state: Dict[str, Any]) -> str:
            repair_type = state.get("repair_type", "")
            approach = state.get("approach", "hybrid")

            if repair_type == "sql_error":
                return "sql_generator"

            # Wrong/invalid output or citations: loop back to plausible culprit
            if repair_type in ["invalid_output", "invalid_citations"]:
                if approach == "rag":
                    return "rag_retriever"
                elif approach == "sql":
                    # Regenerate/refine SQL for better results/citations
                    return "sql_generator"
                else:  # hybrid
                    # Re-run RAG; hybrid flow will subsequently proceed to SQL again
                    return "rag_retriever"

            # Invalid format is a synthesis issue; retry synthesizer with error context
            if repair_type == "invalid_format":
                return "synthesizer"

            # Max repairs reached or unknown error
            return END
        
        graph.add_conditional_edges(
            "repair",
            route_after_repair,
            {
                # "sql_executor": "sql_executor",
                "sql_generator": "sql_generator",
                "rag_retriever": "rag_retriever",
                "synthesizer": "synthesizer",
                END: END
            }
        )
        
        return graph
    
    def _validate_output_format(self, answer: Any, format_hint: str) -> bool:
        """Validate output format."""
        if not format_hint:
            return True
        
        try:
            if format_hint == "int":
                return isinstance(answer, int)
            elif format_hint == "float":
                return isinstance(answer, (int, float))
            elif format_hint.startswith("list[") or format_hint.startswith("List["):
                return isinstance(answer, list)
            elif format_hint.startswith("{") or format_hint == "dict":
                return isinstance(answer, dict)
            else:
                return True
        except:
            return False
    
    def process_question(self, question: str, format_hint: str, question_id: str = "") -> Dict[str, Any]:
        """Process a single question through the hybrid agent."""
        initial_state = {
            "question": question,
            "format_hint": format_hint,
            "id": question_id,
            "repair_count": 0,
            "max_repairs": 2,
            # Pre-populate list/dict fields so reducers (op.add / op.or_) merge correctly across nodes
            "ontological_blocks": [],
            "date_ranges": [],
            "kpi_formulas": [],
            "categories_entities": [],
            "rag_graph_repr": {},
            "rag_search_tags": [],
            "rag_chunks": [],
            "expected_columns": [],
            "sql_results": [],
            "sql_columns": [],
            "citations": [],
            "trace_log": []
        }
        
        # Compile and run the graph
        compiled_graph = self.graph.compile()
        print("Graph compiled")
        
        try:
            print("Graph invoke start")
            final_state = compiled_graph.invoke(initial_state)
            print("Graph invoke end")

            # Minimal guard: if a list is returned, pick the last dict-like state
            if isinstance(final_state, list):
                final_state = next((s for s in reversed(final_state) if isinstance(s, dict)), initial_state)
            
            # Extract final answer with proper SQL citation
            sql_citations = []
            if final_state.get("sql_success") and final_state.get("sql_results"):
                # Add table citations based on SQL query
                sql_query = final_state.get("generated_sql", "").upper()
                tables = ["Orders", "Order Details", "Products", "Customers", "Categories", 
                         "Suppliers", "Employees", "Shippers", "Regions", "Territories"]
                for table in tables:
                    if table.upper() in sql_query:
                        sql_citations.append(table)
            
            # Combine with existing citations
            all_citations = list(set(sql_citations + (final_state.get("citations", []) or [])))
            
            return {
                "id": question_id,
                "final_answer": final_state.get("final_answer"),
                "sql": final_state.get("generated_sql", ""),
                "confidence": final_state.get("confidence", 0.0),
                "explanation": final_state.get("explanation", ""),
                "citations": all_citations,
                "trace_log": final_state.get("trace_log", [])
            }
            
        except Exception as e:
            try:
                _write_error_log(
                    component="HybridAgent.process_question",
                    error=str(e),
                    details={
                        "question_id": question_id,
                        "question": (question[:200] + '...') if len(question) > 200 else question,
                        "format_hint": format_hint,
                    }
                )
            except Exception:
                pass
            return {
                "id": question_id,
                "final_answer": None,
                "sql": "",
                "confidence": 0.0,
                "explanation": f"Processing failed: {str(e)}",
                "citations": [],
                "trace_log": initial_state.get("trace_log", [])
            }

 
