"""DSPy signatures and modules for the retail analytics copilot.

Refactor: avoid DSPy structured outputs. Build JSON schemas in prompts and parse JSON responses.
"""

import dspy
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from pathlib import Path
import datetime
import json
import re


class RequirementParserInput(BaseModel):
    """Input for the requirement parser."""
    question: str
    format_hint: str


class RequirementParserOutput(BaseModel):
    """Output from the requirement parser."""
    ontological_blocks: List[Dict[str, Any]] = Field(description="List of required information blocks with tags")
    expected_answer_type: str = Field(description="Expected type of the final answer")
    date_ranges: List[str] = Field(default_factory=list, description="Date ranges mentioned in the question")
    kpi_formulas: List[str] = Field(default_factory=list, description="KPI formulas needed")
    categories_entities: List[Dict[str, Any]] = Field(default_factory=list, description="Categories and entities mentioned")


class RouterInput(BaseModel):
    """Input for the router classifier."""
    question: str
    ontological_blocks: List[Dict[str, Any]]
    expected_answer_type: str


class RouterOutput(BaseModel):
    """Output from the router classifier."""
    approach: str = Field(description="One of: rag, sql, hybrid")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Brief explanation of the routing decision")


class SQLGeneratorInput(BaseModel):
    """Input for the SQL generator."""
    question: str
    schema_info: str
    ontological_blocks: List[Dict[str, Any]]
    previous_sql: Optional[str] = None
    error_message: Optional[str] = None


class SQLGeneratorOutput(BaseModel):
    """Output from the SQL generator."""
    sql_query: str = Field(description="Generated SQL query")
    explanation: str = Field(description="Brief explanation of the query logic")
    expected_columns: List[str] = Field(description="Expected column names in the result")


class SynthesizerInput(BaseModel):
    """Input for the synthesizer."""
    question: str
    format_hint: str
    sql_results: Optional[List[Dict[str, Any]]] = None
    rag_chunks: Optional[List[Dict[str, Any]]] = None
    ontological_blocks: List[Dict[str, Any]]
    previous_attempt: Optional[str] = None
    error_message: Optional[str] = None


class SynthesizerOutput(BaseModel):
    """Output from the synthesizer."""
    final_answer: Any = Field(description="Final answer matching the format_hint")
    confidence: float = Field(description="Confidence score between 0 and 1")
    explanation: str = Field(description="Brief explanation (<=2 sentences)")
    citations: List[str] = Field(description="List of citations (DB tables and doc chunk IDs)")


# DSPy Signatures
class RequirementParserSignature(dspy.Signature):
    """Parse natural language question into ontological blocks and extract constraints."""
    question: str = dspy.InputField(desc="Natural language question")
    format_hint: str = dspy.InputField(desc="Expected format of the answer")
    
    ontological_blocks: str = dspy.OutputField(desc="JSON string of required information blocks with tags (RAG/SQL/both)")
    expected_answer_type: str = dspy.OutputField(desc="Expected type of final answer")
    date_ranges: str = dspy.OutputField(desc="JSON list of date ranges mentioned")
    kpi_formulas: str = dspy.OutputField(desc="JSON list of KPI formulas needed")
    categories_entities: str = dspy.OutputField(desc="JSON list of categories and entities")


class RouterSignature(dspy.Signature):
    """Route the question to appropriate approach: rag, sql, or hybrid."""
    question: str = dspy.InputField(desc="Natural language question")
    ontological_blocks: str = dspy.InputField(desc="JSON string of ontological blocks")
    expected_answer_type: str = dspy.InputField(desc="Expected answer type")
    
    approach: str = dspy.OutputField(desc="One of: rag, sql, hybrid")
    confidence: float = dspy.OutputField(desc="Confidence score 0-1")
    reasoning: str = dspy.OutputField(desc="Brief routing decision explanation")


class SQLGeneratorSignature(dspy.Signature):
    """Generate SQLite query from natural language question and schema."""
    question: str = dspy.InputField(desc="Natural language question")
    schema_info: str = dspy.InputField(desc="Database schema information")
    ontological_blocks: str = dspy.InputField(desc="JSON string of ontological blocks")
    previous_sql: str = dspy.InputField(desc="Previous SQL attempt (if repair)", default="")
    error_message: str = dspy.InputField(desc="Error message from previous attempt (if repair)", default="")
    
    sql_query: str = dspy.OutputField(desc="Generated SQLite query")
    explanation: str = dspy.OutputField(desc="Brief query explanation")
    expected_columns: str = dspy.OutputField(desc="JSON list of expected column names")


class SynthesizerSignature(dspy.Signature):
    """Synthesize final answer from SQL results and/or RAG chunks."""
    question: str = dspy.InputField(desc="Original natural language question")
    format_hint: str = dspy.InputField(desc="Required format of final answer")
    sql_results: str = dspy.InputField(desc="JSON string of SQL query results", default="")
    rag_chunks: str = dspy.InputField(desc="JSON string of retrieved document chunks", default="")
    ontological_blocks: str = dspy.InputField(desc="JSON string of ontological blocks")
    previous_attempt: str = dspy.InputField(desc="Previous synthesis attempt (if repair)", default="")
    error_message: str = dspy.InputField(desc="Error message from previous attempt (if repair)", default="")
    
    final_answer: str = dspy.OutputField(desc="Final answer matching format_hint exactly")
    confidence: float = dspy.OutputField(desc="Confidence score 0-1")
    explanation: str = dspy.OutputField(desc="Brief explanation (<=2 sentences)")
    citations: str = dspy.OutputField(desc="JSON list of citations (DB tables and doc chunk IDs)")


# DSPy Modules
class RequirementParser(dspy.Module):
    """Parse requirements from natural language questions."""
    
    def __init__(self):
        super().__init__()
        self.parse = dspy.ChainOfThought(RequirementParserSignature)  # retained but unused
    
    def _ensure_logs_dir(self) -> Path:
        base_dir = Path(__file__).resolve().parents[1]
        logs_dir = base_dir / 'llm_logs'
        try:
            logs_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return logs_dir
    
    def _write_llm_log(self, component: str, prompt: Dict[str, Any], response: Dict[str, Any]) -> None:
        logs_dir = self._ensure_logs_dir()
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = logs_dir / f"{component}_{ts}.log"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(json.dumps({
                    "timestamp": ts,
                    "component": component,
                    "prompt": prompt,
                    "response": response
                }, ensure_ascii=False, indent=2))
        except Exception:
            pass
    
    def _extract_json(self, text: Any) -> Dict[str, Any]:
        # Already parsed structures
        if isinstance(text, dict):
            return text
        if isinstance(text, list):
            # Try to parse the first parsable item
            for item in text:
                parsed = self._extract_json(item)
                if parsed:
                    return parsed
            return {}
        # Fallback to string-based parsing
        if not isinstance(text, str):
            text = str(text)
        # Direct load if valid JSON string
        try:
            return json.loads(text)
        except Exception:
            pass
        # Try ALL fenced code blocks (with or without json language tag)
        for block_any in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE):
            try:
                return json.loads(block_any.group(1).strip())
            except Exception:
                continue
        # Try to find JSON objects in the text (non-greedy, iterate all)
        for match in re.finditer(r"\{[\s\S]*?\}", text):
            candidate = match.group(0)
            try:
                return json.loads(candidate)
            except Exception:
                continue
        # Try to find a JSON array in the text
        arr = re.search(r"\[[\s\S]*\]", text)
        if arr:
            try:
                return json.loads(arr.group(0))
            except Exception:
                pass
        # Fallback: extract pruned_schema value even if overall JSON is malformed
        try:
            m = re.search(r'"pruned_schema"\s*:\s*(?:"""|\'\'\'|"|\')([\s\S]*?)(?:"""|\'\'\'|"|\')', text)
            if m:
                return {"pruned_schema": m.group(1).strip()}
        except Exception:
            pass
        return {}
    
    def _load_text_file(self, filename: str) -> str:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _db_diagram(self) -> str:
        return self._load_text_file("/home/mohammed/Desktop/tech_projects/northwind_ai_workflow/agent/sql/northwind_dbdiagram.txt")
    
    def _docs_overview(self) -> str:
        base_dir = Path(__file__).resolve().parents[1]
        docs_dir = base_dir / 'docs'
        parts: List[str] = []
        try:
            for path in sorted(docs_dir.iterdir(), key=lambda p: p.name.lower()):
                if path.is_file():
                    try:
                        content = self._load_text_file(str(path))
                    except Exception:
                        content = ""
                    parts.append(f"File: {path.name}\n{content}")
        except Exception:
            pass
        return "\n\n---\n\n".join(parts)

    
    def _lm_json(self, title: str, instruction: str, inputs: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            f"{title}\n" 
            f"{instruction}\n\n"
            f"Inputs:\n{json.dumps(inputs, ensure_ascii=False, indent=2)}\n\n"
            f"Return JSON ONLY matching this schema (keys and types):\n"
            f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n"
        )
        try:
            raw = dspy.settings.lm(prompt)
        except Exception as e:
            raw = str(e)
        data = self._extract_json(raw)
        try:
            self._write_llm_log(component=title.replace(' ', ''), prompt={"instruction": instruction, **inputs, "schema": schema}, response={"raw": raw, "parsed": data})
        except Exception:
            pass
        return data
    
    def forward(self, question: str, format_hint: str) -> RequirementParserOutput:
        """Parse the question into ontological blocks using JSON-prompting."""
        schema = {
            "ontological_blocks": [
                {"content": "string", "tags": ["string"]}
            ],
            "expected_answer_type": "string",
            "date_ranges": ["string"],
            "kpi_formulas": ["string"],
            "categories_entities": [{"content": "string", "tags": ["string"]}]
        }

        db_diagram = self._db_diagram()
        docs_overview = self._docs_overview()
        
        data = self._lm_json(
            title="RequirementParser",
            instruction="Extract useful ontological blocks to achieve the required answer from the database and documents, expected answer type, and constraints.",
            inputs={"question": question, "format_hint": format_hint, "db_diagram": db_diagram, "docs_overview": docs_overview, },
            schema=schema,
        )
        ontological_blocks = data.get("ontological_blocks") or []
        if not isinstance(ontological_blocks, list):
            ontological_blocks = [{"content": str(ontological_blocks), "tags": ["hybrid"]}]
        expected_answer_type = data.get("expected_answer_type") or data.get("expected_answer") or "string"
        date_ranges = data.get("date_ranges") or []
        if not isinstance(date_ranges, list):
            date_ranges = []
        kpi_formulas = data.get("kpi_formulas") or []
        if not isinstance(kpi_formulas, list):
            kpi_formulas = []
        categories_entities = data.get("categories_entities") or []
        if not isinstance(categories_entities, list):
            categories_entities = []
        # Coerce categories_entities to list of dicts if model returned strings
        coerced_entities = []
        for item in categories_entities:
            if isinstance(item, dict):
                coerced_entities.append(item)
            else:
                coerced_entities.append({"content": str(item), "tags": []})
        categories_entities = coerced_entities
        return RequirementParserOutput(
            ontological_blocks=ontological_blocks,
            expected_answer_type=expected_answer_type,
            date_ranges=date_ranges,
            kpi_formulas=kpi_formulas,
            categories_entities=categories_entities,
        )


class Router(dspy.Module):
    """Route questions to appropriate processing approach."""
    
    def __init__(self):
        super().__init__()
        self.route = dspy.ChainOfThought(RouterSignature)  # retained but unused
        self._logger = RequirementParser()
    
    def forward(self, question: str, ontological_blocks: List[Dict[str, Any]], 
                expected_answer_type: str) -> RouterOutput:
        """Route the question to the appropriate approach using JSON-prompting."""
        schema = {
            "approach": "string (rag|sql|hybrid)",
            "confidence": 0.0,
            "reasoning": "string"
        }
        data = self._logger._lm_json(
            title="Router",
            instruction="Choose processing approach.",
            inputs={
                "question": question,
                "ontological_blocks": ontological_blocks,
                "expected_answer_type": expected_answer_type
            },
            schema=schema,
        )
        approach_raw = (data.get("approach") or "").lower().strip()
        confidence = data.get("confidence")
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.8
        reasoning = data.get("reasoning") or ""
        
        # Ensure approach is valid
        approach = approach_raw
        if approach not in ["rag", "sql", "hybrid"]:
            # Default routing logic
            if any("date" in str(block).lower() or "revenue" in str(block).lower() 
                   for block in ontological_blocks):
                approach = "hybrid" if any("policy" in str(block).lower() or "kpi" in str(block).lower() 
                                        for block in ontological_blocks) else "sql"
            else:
                approach = "rag"
        
        return RouterOutput(
            approach=approach,
            confidence=confidence,
            reasoning=reasoning
        )


class SQLGenerator(dspy.Module):
    """Generate SQL queries from natural language."""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(SQLGeneratorSignature)  # retained but unused
        self._logger = RequirementParser()
    
    def forward(self, question: str, schema_info: str, ontological_blocks: List[Dict[str, Any]],
                previous_sql: Optional[str] = None, error_message: Optional[str] = None) -> SQLGeneratorOutput:
        """Generate SQL query using JSON-prompting."""
        schema = {
            "sql_query": "string",
            "explanation": "string",
            "expected_columns": ["string"]
        }
        instruction = (
            "Use ONLY schema_info. No new tables/columns. Use exact quoted names from schema_info including spaces. Return valid SQLite only. "
            "For table names with spaces use quotes like Order Details table name. Never invent columns or aliases. Prefer strftime for year extraction. "
            "expected_columns must match SELECT aliases exactly."
        )
        data = self._logger._lm_json(
            title="SQLGenerator",
            instruction=instruction,
            inputs={
                "question": question,
                "schema_info": schema_info,
                "ontological_blocks": ontological_blocks,
                "previous_sql": previous_sql or "",
                "error_message": error_message or ""
            },
            schema=schema,
        )
        # Minimal coercion: tolerate accidental list and try to recover
        if not isinstance(data, dict):
            if isinstance(data, list):
                # try to find a dict inside
                inner = next((x for x in data if isinstance(x, dict)), None)
                data = inner or {}
            else:
                data = {}
        sql_query = (data.get("sql_query") or "").strip()
        explanation = data.get("explanation") or ""
        expected_columns = data.get("expected_columns") or []
        if not isinstance(expected_columns, list):
            expected_columns = []
        return SQLGeneratorOutput(
            sql_query=sql_query,
            explanation=explanation,
            expected_columns=expected_columns
        )


class Synthesizer(dspy.Module):
    """Synthesize final answers from multiple sources."""
    
    def __init__(self):
        super().__init__()
        self.synthesize = dspy.ChainOfThought(SynthesizerSignature)  # retained but unused
        self._logger = RequirementParser()
    
    def forward(self, question: str, format_hint: str, 
                sql_results: Optional[List[Dict[str, Any]]] = None,
                rag_chunks: Optional[List[Dict[str, Any]]] = None,
                ontological_blocks: List[Dict[str, Any]] = None,
                previous_attempt: Optional[str] = None,
                error_message: Optional[str] = None) -> SynthesizerOutput:
        """Synthesize final answer using JSON-prompting."""
        schema = {
            "final_answer": "(type must match format_hint)",
            "confidence": 0.0,
            "explanation": "string (<=2 sentences)",
            "citations": ["string"]
        }
        # Enhanced instruction for better handling of missing data
        instruction = (
            "Combine inputs to produce final answer in the required format and include citations. "
            "If sql_results is empty but question requires data analysis, explain that data could not be retrieved. "
            "Use rag_chunks for context when available. Always provide a helpful response even with limited data."
        )
        data = self._logger._lm_json(
            title="Synthesizer",
            instruction=instruction,
            inputs={
                "question": question,
                "format_hint": format_hint,
                "sql_results": sql_results or [],
                "rag_chunks": rag_chunks or [],
                "ontological_blocks": ontological_blocks or [],
                "previous_attempt": previous_attempt or "",
                "error_message": error_message or ""
            },
            schema=schema,
        )
        citations = data.get("citations") or []
        if not isinstance(citations, list):
            citations = []
        final_answer = data.get("final_answer")
        # Parse based on format_hint
        try:
            if format_hint.startswith("list[") or format_hint.startswith("List["):
                if isinstance(final_answer, str):
                    final_answer = json.loads(final_answer)
            elif format_hint.startswith("{") or format_hint == "dict":
                if isinstance(final_answer, str):
                    final_answer = json.loads(final_answer)
            elif format_hint == "int":
                if isinstance(final_answer, str):
                    final_answer = int(float(final_answer.strip()))
            elif format_hint == "float":
                if isinstance(final_answer, str):
                    final_answer = float(final_answer.strip())
        except Exception:
            pass
        confidence = data.get("confidence")
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.8
        explanation = data.get("explanation") or ""
        return SynthesizerOutput(
            final_answer=final_answer,
            confidence=confidence,
            explanation=explanation,
            citations=citations
        )


class SchemaPruner(dspy.Module):
    """LLM-driven schema pruner that returns a minimal DB diagram subset."""
    
    def __init__(self):
        super().__init__()
        self._logger = RequirementParser()
    
    def _db_diagram(self) -> str:
        # Reuse the same diagram source as RequirementParser
        return self._logger._db_diagram()
    
    def forward(self, ontological_blocks: List[Dict[str, Any]]) -> str:
        """Return a pruned schema representation as a string."""
        schema = {
            "pruned_schema": "string"
        }
        db_diagram = self._db_diagram()
        instruction = (
            "Return JSON only: {\"pruned_schema\": \"...\"}. No code fences. "
            "Inside pruned_schema, write a minimal diagram in the SAME format as db_diagram (Table \"...\" { ... } and Ref lines) using exact quoted names. "
            "CRITICAL: Preserve table names with spaces like Order Details exactly as shown in db_diagram. "
            "Avoid over-pruning: include not only the directly mentioned tables/columns in ontological_blocks, but also closely related tables (1-2 hops via foreign keys) that are likely needed for joins, filters (especially dates), or aggregations. "
            "Include all relevant Ref lines among the included tables. Do NOT invent new tables or columns. Prefer slightly broader coverage over omission when uncertain."
        )
        data = self._logger._lm_json(
            title="SchemaPruner",
            instruction=instruction,
            inputs={
                "db_diagram": db_diagram,
                "ontological_blocks": ontological_blocks,
            },
            schema=schema,
        )
        raw_ps = data.get("pruned_schema")
        # Normalize pruned schema to a clean diagram string
        pruned_schema = ""
        if isinstance(raw_ps, list):
            pruned_schema = "\n".join(str(x) for x in raw_ps)
        else:
            pruned_schema = str(raw_ps or "").strip()
            # Attempt to decode if the model double-encoded JSON
            for _ in range(2):
                try:
                    if pruned_schema and (pruned_schema.startswith("[") or pruned_schema.startswith("{")):
                        decoded = json.loads(pruned_schema)
                        if isinstance(decoded, list):
                            pruned_schema = "\n".join(str(x) for x in decoded)
                        elif isinstance(decoded, str):
                            pruned_schema = decoded
                        else:
                            break
                    else:
                        break
                except Exception:
                    break
            pruned_schema = pruned_schema.replace("\\n", "\n").replace("\\\"", "\"")
        return pruned_schema
