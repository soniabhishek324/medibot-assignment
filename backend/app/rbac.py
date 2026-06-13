from __future__ import annotations

from dataclasses import dataclass


ROLES = ("doctor", "nurse", "billing_executive", "technician", "admin")

ROLE_COLLECTIONS: dict[str, list[str]] = {
    "doctor": ["clinical", "nursing", "general"],
    "nurse": ["nursing", "general"],
    "billing_executive": ["billing", "general"],
    "technician": ["equipment", "general"],
    "admin": ["general", "clinical", "nursing", "billing", "equipment"],
}

COLLECTION_ROLES: dict[str, list[str]] = {
    "general": list(ROLES),
    "clinical": ["doctor", "admin"],
    "nursing": ["nurse", "doctor", "admin"],
    "billing": ["billing_executive", "admin"],
    "equipment": ["technician", "admin"],
}

DEMO_USERS = {
    "abhishek.soni": {"password": "demo123", "role": "doctor", "name": "Abhishek Soni"},
    "swati": {"password": "demo123", "role": "nurse", "name": "Swati"},
    "billing.ravi": {"password": "demo123", "role": "billing_executive", "name": "Ravi Menon"},
    "tech.anand": {"password": "demo123", "role": "technician", "name": "Anand Rao"},
    "admin.sys": {"password": "demo123", "role": "admin", "name": "System Admin"},
}


@dataclass(frozen=True)
class UserSession:
    username: str
    name: str
    role: str


def collections_for_role(role: str) -> list[str]:
    return ROLE_COLLECTIONS.get(role, [])


def can_access_collection(role: str, collection: str) -> bool:
    return collection in collections_for_role(role)


def rbac_refusal(role: str, requested: str | None = None) -> str:
    allowed = ", ".join(collections_for_role(role))
    if requested:
        return f"As a {role}, you do not have access to {requested} documents. I can only answer from these collections: {allowed}."
    return f"As a {role}, I can only answer from these collections: {allowed}."
