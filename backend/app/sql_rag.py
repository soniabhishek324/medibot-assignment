from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from app.config import get_settings
from app.llm import generate_with_llm


SCHEMA = """
claims(claim_id, patient_id, patient_name, department, claim_type, diagnosis_code,
insurer, claimed_amount, approved_amount, status, submitted_date, resolved_date)
maintenance_tickets(ticket_id, equipment_name, equipment_id, category, campus,
issue_type, fault_code, raised_by, raised_date, resolved_date, status, resolution_note)
"""


def is_analytical_question(question: str) -> bool:
    terms = ["how many", "count", "total", "average", "sum", "most", "least", "open", "approved", "rejected", "escalated", "pending"]
    return any(term in question.lower() for term in terms)


def clean_sql(raw: str) -> str:
    match = re.search(r"```sql\s*(.*?)```", raw, re.IGNORECASE | re.DOTALL) or re.search(r"```\s*(.*?)```", raw, re.DOTALL)
    candidate = match.group(1) if match else raw
    sql_match = re.search(r"\bSELECT\b[\s\S]*?(?:;|$)", candidate, re.IGNORECASE)
    if not sql_match:
        raise ValueError("Only SELECT statements are allowed.")
    sql = sql_match.group(0).strip().rstrip(";")
    if not re.match(r"^SELECT\b", sql, re.IGNORECASE):
        raise ValueError("Only SELECT statements are allowed.")
    forbidden = re.search(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|PRAGMA|ATTACH)\b", sql, re.IGNORECASE)
    if forbidden:
        raise ValueError("Unsafe SQL statement rejected.")
    return sql


def fallback_sql(question: str) -> str:
    q = question.lower()
    if "maintenance" in q or "ticket" in q or "equipment" in q:
        if "most" in q and "open" in q:
            return "SELECT category, COUNT(*) AS open_tickets FROM maintenance_tickets WHERE status IN ('open','in_progress','escalated') GROUP BY category ORDER BY open_tickets DESC LIMIT 5"
        if "escalated" in q:
            return "SELECT COUNT(*) AS escalated_tickets FROM maintenance_tickets WHERE status = 'escalated'"
        if "open" in q or "pending" in q:
            return "SELECT status, COUNT(*) AS tickets FROM maintenance_tickets WHERE status IN ('open','in_progress','escalated') GROUP BY status"
        return "SELECT category, COUNT(*) AS tickets FROM maintenance_tickets GROUP BY category ORDER BY tickets DESC LIMIT 5"
    if "approved" in q:
        return "SELECT COUNT(*) AS approved_claims, SUM(approved_amount) AS approved_amount FROM claims WHERE status = 'approved'"
    if "rejected" in q:
        return "SELECT COUNT(*) AS rejected_claims FROM claims WHERE status = 'rejected'"
    if "pending" in q or "open" in q:
        return "SELECT COUNT(*) AS pending_claims, SUM(claimed_amount) AS claimed_amount FROM claims WHERE status = 'pending'"
    if "total" in q or "sum" in q:
        return "SELECT SUM(claimed_amount) AS total_claimed_amount FROM claims"
    return "SELECT status, COUNT(*) AS claims, SUM(claimed_amount) AS claimed_amount FROM claims GROUP BY status ORDER BY claims DESC"


def execute_sql(sql: str, db_path: Path) -> tuple[list[str], list[tuple]]:
    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(sql)
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
    return columns, rows


def sql_rag_chain(question: str) -> str:
    settings = get_settings()
    raw_sql = generate_with_llm(
        "You write safe SQLite SELECT queries only.",
        f"Schema:\n{SCHEMA}\nQuestion: {question}\nReturn only one SQLite SELECT statement.",
    )
    sql = clean_sql(raw_sql or fallback_sql(question))
    columns, rows = execute_sql(sql, settings.sqlite_db_path)
    llm_answer = generate_with_llm(
        "You explain database query results for hospital operations staff.",
        f"Question: {question}\nSQL: {sql}\nColumns: {columns}\nRows: {rows}\nAnswer concisely.",
    )
    if llm_answer:
        return llm_answer
    if not rows:
        return f"I ran `{sql}` and found no matching records."
    rendered_rows = ["; ".join(f"{column}: {value}" for column, value in zip(columns, row)) for row in rows[:8]]
    return f"I ran `{sql}`. Results: " + " | ".join(rendered_rows)
