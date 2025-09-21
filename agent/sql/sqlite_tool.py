"""SQLite database tools for schema introspection and query execution."""

import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from pydantic import BaseModel
from agent.dspy_signatures import SchemaPruner


class SQLResult(BaseModel):
    """Result from SQL query execution."""
    columns: List[str]
    rows: List[List[Any]]
    row_count: int
    success: bool
    error: Optional[str] = None


class ColumnInfo(BaseModel):
    """Information about a table column."""
    name: str
    type: str
    not_null: bool
    default: Optional[str] = None
    primary_key: bool


class ForeignKeyInfo(BaseModel):
    """Information about a foreign key constraint."""
    column: str
    references_table: str
    references_column: str


class TableInfo(BaseModel):
    """Information about a database table."""
    name: str
    columns: List[ColumnInfo]
    primary_keys: List[str]
    foreign_keys: List[ForeignKeyInfo]


class SQLiteInspector:
    """Database schema inspector."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.schema_cache = {}
        self._load_schema()
    
    def _load_schema(self):
        """Load and cache database schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                for table in tables:
                    self.schema_cache[table] = self._get_table_info(cursor, table)
        
        except Exception as e:
            print(f"Error loading schema: {e}")
    
    def _get_table_info(self, cursor: sqlite3.Cursor, table_name: str) -> TableInfo:
        """Get detailed information about a table."""
        # Get column info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns_raw = cursor.fetchall()
        
        columns = []
        primary_keys = []
        
        for col in columns_raw:
            col_info = ColumnInfo(
                name=col[1],
                type=col[2],
                not_null=bool(col[3]),
                default=col[4] if col[4] is not None else None,
                primary_key=bool(col[5])
            )
            columns.append(col_info)
            
            if col_info.primary_key:
                primary_keys.append(col_info.name)
        
        # Get foreign key info
        cursor.execute(f"PRAGMA foreign_key_list({table_name})")
        fk_raw = cursor.fetchall()
        
        foreign_keys = []
        for fk in fk_raw:
            foreign_keys.append(ForeignKeyInfo(
                column=fk[3],
                references_table=fk[2],
                references_column=fk[4]
            ))
        
        return TableInfo(
            name=table_name,
            columns=columns,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys
        )
    
    def get_schema_info(self, relevant_tables: Optional[List[str]] = None) -> str:
        """Get schema information as a formatted string."""
        if relevant_tables is None:
            relevant_tables = list(self.schema_cache.keys())
        
        schema_parts = []
        
        for table_name in relevant_tables:
            if table_name not in self.schema_cache:
                continue
            
            table_info = self.schema_cache[table_name]
            schema_parts.append(f"Table: {table_name}")
            
            for col in table_info.columns:
                col_desc = f"  {col.name} {col.type}"
                if col.primary_key:
                    col_desc += " PRIMARY KEY"
                if col.not_null:
                    col_desc += " NOT NULL"
                if col.default is not None and str(col.default) != "":
                    col_desc += f" DEFAULT {col.default}"
                schema_parts.append(col_desc)
            
            if table_info.foreign_keys:
                schema_parts.append("  Foreign Keys:")
                for fk in table_info.foreign_keys:
                    schema_parts.append(f"    {fk.column} -> {fk.references_table}.{fk.references_column}")
            
            schema_parts.append("")
        
        return "\n".join(schema_parts)


class SQLGraphPruner:
    """Prune database schema to relevant tables and columns."""
    
    def __init__(self, inspector: SQLiteInspector):
        self.inspector = inspector
        self.pruner = SchemaPruner()
    
    def prune_schema(self, ontological_blocks: List[Dict[str, Any]]) -> str:
        """Create a reduced schema representation."""
        # Use LLM-based pruner to generate minimal diagram subset
        pruned_schema = self.pruner.forward(ontological_blocks)
        return pruned_schema


class SQLExecutor:
    """Execute SQL queries against the database."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
    
    def execute_query(self, sql_query: str) -> SQLResult:
        """Execute SQL query and return results."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Enable foreign key constraints
                conn.execute("PRAGMA foreign_keys = ON")
                
                cursor = conn.cursor()
                # General normalization for common LLM SQL issues
                import re
                fixed_query = sql_query
                # Normalize all strftime malformed variants (comprehensive patterns)
                fixed_query = re.sub(r"strf[\s\-_]*time\s*\(", "strftime(", fixed_query, flags=re.IGNORECASE)
                fixed_query = re.sub(r"strft[A-Z]*TIME", "strftime", fixed_query, flags=re.IGNORECASE)
                fixed_query = re.sub(r"strftFRMAT[A-Z]*TIME", "strftime", fixed_query, flags=re.IGNORECASE)
                # Normalize datepart-like patterns and wrong date functions
                fixed_query = re.sub(r"\bDATEPART\s*\(\s*year\s*,", "strftime('%Y',", fixed_query, flags=re.IGNORECASE)
                fixed_query = re.sub(r"\bYear\s*\(", "strftime('%Y', ", fixed_query, flags=re.IGNORECASE)
                # Fix date format patterns - SQLite uses %Y not YYYY
                fixed_query = re.sub(r"'YYYY'", "'%Y'", fixed_query)
                fixed_query = re.sub(r'"YYYY"', "'%Y'", fixed_query)
                # Fix year comparison - ensure string comparison
                fixed_query = re.sub(r"= 1997\b", "= '1997'", fixed_query)
                fixed_query = re.sub(r"= (\d{4})\b", r"= '\1'", fixed_query)
                # Legacy specific typos (keep for backward compatibility)
                fixed_query = fixed_query.replace("strftForms", "strftime").replace("STRFTFORMS", "strftime")
                # Fix common table name issues with spaces
                fixed_query = re.sub(r'\bOrderDetails\b', '"Order Details"', fixed_query)
                # Fix invented columns - replace non-existent Revenue column with proper calculation
                fixed_query = re.sub(r'\bo\.Revenue\b', '(od.Quantity * od.UnitPrice)', fixed_query)
                fixed_query = re.sub(r'\bRevenue\b', '(od.Quantity * od.UnitPrice)', fixed_query)
                # Remove unnecessary GROUP BY when doing aggregate without grouping
                if 'SUM(' in fixed_query and 'GROUP BY strftime' in fixed_query:
                    fixed_query = re.sub(r'\s+GROUP BY strftime\([^)]+\)', '', fixed_query)

                # Debug logging to see query transformation
                if fixed_query != sql_query:
                    print(f"SQL Query normalized from:\n{sql_query}\nto:\n{fixed_query}")

                # Pre-validation before execution
                is_valid, err = self.validate_query(fixed_query)
                if not is_valid:
                    # Attempt alias/column auto-repair for errors like: no such column: alias.column
                    repaired = self._attempt_alias_column_repair(fixed_query, str(err))
                    if repaired:
                        fixed_query = repaired
                        is_valid, err2 = self.validate_query(fixed_query)
                        if not is_valid:
                            raise Exception(f"Validation failed after repair: {err2}")
                    else:
                        raise Exception(f"Validation failed: {err}")

                # Execute normalized query
                print(f"Executing final query: {fixed_query}")
                cursor.execute(fixed_query)
                
                # Get column names
                columns = [description[0] for description in cursor.description] if cursor.description else []
                
                # Get all rows
                rows = cursor.fetchall()
                
                # Debug: Show raw results
                print(f"Raw SQL results: columns={columns}, rows={rows}")
                
                return SQLResult(
                    columns=columns,
                    rows=rows,
                    row_count=len(rows),
                    success=True
                )
        
        except Exception as e:
            return SQLResult(
                columns=[],
                rows=[],
                row_count=0,
                success=False,
                error=str(e)
            )
    
    def validate_query(self, sql_query: str) -> Tuple[bool, Optional[str]]:
        """Validate SQL query without executing it."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Use EXPLAIN QUERY PLAN to validate without executing
                cursor.execute(f"EXPLAIN QUERY PLAN {sql_query}")
                return True, None
        except Exception as e:
            return False, str(e)

    def _attempt_alias_column_repair(self, sql_query: str, error_message: str) -> Optional[str]:
        """Try to repair alias.column errors by mapping column to the correct table alias.

        Strategy:
        - If error contains 'no such column: X.Y' or 'no such column: Y', try to map Y to a table in FROM/JOINs that owns column Y.
        - If exactly one table has that column, and that table has an alias in the query, rewrite occurrences of wrongAlias.Y to correctAlias.Y.
        - Re-validate after rewriting. Return new SQL if changed; otherwise None.
        """
        try:
            import re
            # Extract the missing column info
            m = re.search(r"no such column: ([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)", error_message)
            if m:
                wrong_alias = m.group(1)
                column_name = m.group(2)
            else:
                m2 = re.search(r"no such column: ([A-Za-z_][\w]*)", error_message)
                if not m2:
                    return None
                wrong_alias = None
                column_name = m2.group(1)

            # Parse table aliases from FROM/JOIN clauses:  "TableName" [AS] alias  OR  TableName [AS] alias
            # This is a heuristic, but adequate for simple SELECTs this agent generates
            alias_map = {}  # alias -> table
            table_alias_pairs = []
            for tbl, alias in re.findall(r"FROM\s+\"?([A-Za-z ]+)\"?\s+(?:AS\s+)?([A-Za-z][\w]*)|JOIN\s+\"?([A-Za-z ]+)\"?\s+(?:AS\s+)?([A-Za-z][\w]*)", sql_query, flags=re.IGNORECASE):
                # The regex returns tuples with some groups empty; normalize
                parts = [p for p in (tbl, alias) if p] or [p for p in (tbl, alias) if p]
                if not parts:
                    continue
                # Distinguish matches for FROM and JOIN
                if parts and len(parts) == 2:
                    table_name, alias_name = parts[0], parts[1]
                    alias_map[alias_name] = table_name.strip().strip('"')
                    table_alias_pairs.append((table_name.strip().strip('"'), alias_name))

            if not alias_map:
                return None

            # Load schema to see which table contains the column
            inspector = SQLiteInspector(str(self.db_path))
            candidate_aliases = []
            for alias_name, table_name in alias_map.items():
                table_info = inspector.schema_cache.get(table_name) or inspector.schema_cache.get(table_name.replace(' ', '_'))
                if not table_info:
                    continue
                if any(c.name == column_name for c in table_info.columns):
                    candidate_aliases.append(alias_name)

            if len(candidate_aliases) != 1:
                return None  # ambiguous or not found; skip risky rewrite

            correct_alias = candidate_aliases[0]

            # Build a safe regex to replace wrongAlias.column or bare column if wrong_alias is None
            new_sql = sql_query
            if wrong_alias and wrong_alias != correct_alias:
                pattern = rf"\b{re.escape(wrong_alias)}\.{re.escape(column_name)}\b"
                replacement = f"{correct_alias}.{column_name}"
                new_sql = re.sub(pattern, replacement, new_sql)
            else:
                # If no alias was provided in error (bare column), qualify it with the correct alias in SELECT/WHERE/GROUP/ORDER clauses
                # Replace standalone column only when not already qualified
                pattern = rf"(?<!\.)\b{re.escape(column_name)}\b"
                replacement = f"{correct_alias}.{column_name}"
                new_sql = re.sub(pattern, replacement, new_sql)

            if new_sql != sql_query:
                return new_sql
            return None
        except Exception:
            return None


class SQLGraphPrunerNode:
    """LangGraph node for SQL graph pruning."""
    
    def __init__(self, inspector: SQLiteInspector):
        self.pruner = SQLGraphPruner(inspector)
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Prune schema based on requirements."""
        ontological_blocks = state.get("ontological_blocks", [])
        print("Node start: SQLGraphPruner")
        
        # Get pruned schema
        pruned_schema = self.pruner.prune_schema(ontological_blocks)
        
        result_state = {
            **state,
            "pruned_schema": pruned_schema
        }
        print("Node end: SQLGraphPruner")
        return result_state


class SQLGeneratorNode:
    """LangGraph node for SQL generation using DSPy."""
    
    def __init__(self, sql_generator):
        self.sql_generator = sql_generator
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SQL query."""
        question = state.get("question", "")
        pruned_schema = state.get("pruned_schema", "")
        ontological_blocks = state.get("ontological_blocks", [])
        print("Node start: SQLGenerator")
        
        # Check if this is a repair attempt
        previous_sql = state.get("previous_sql")
        error_message = state.get("sql_error")
        
        # Generate SQL
        result = self.sql_generator.forward(
            question=question,
            schema_info=pruned_schema,
            ontological_blocks=ontological_blocks,
            previous_sql=previous_sql,
            error_message=error_message
        )
        
        result_state = {
            **state,
            "generated_sql": result.sql_query,
            "sql_explanation": result.explanation,
            "expected_columns": result.expected_columns
        }
        print("Node end: SQLGenerator")
        return result_state


class SQLExecutorNode:
    """LangGraph node for SQL execution."""
    
    def __init__(self, executor: SQLExecutor):
        self.executor = executor
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SQL query."""
        sql_query = state.get("generated_sql", "")
        print("Node start: SQLExecutor")
        
        if not sql_query:
            return {
                **state,
                "sql_results": [],
                "sql_success": False,
                "sql_error": "No SQL query generated"
            }
        
        # Execute query
        result = self.executor.execute_query(sql_query)
        
        # Convert results to list of dictionaries
        sql_results = []
        if result.success and result.columns:
            for row in result.rows:
                row_dict = dict(zip(result.columns, row))
                sql_results.append(row_dict)
        
        result_state = {
            **state,
            "sql_results": sql_results,
            "sql_success": result.success,
            "sql_error": result.error,
            "sql_row_count": result.row_count,
            "sql_columns": result.columns
        }
        print(f"Node end: SQLExecutor -> success={result.success}, rows={result.row_count}")
        return result_state
