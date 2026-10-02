# -*- coding: utf-8 -*-
"""
api/dependencies.py

FastAPI dependency injection utilities for ProcurementQA_Agent.
Provides singleton access to ProcurementService and tenant extraction.
"""

import os
import sys
from typing import Optional
from fastapi import Header, Request

from core.service import ProcurementService
from core.utils.settings import env

_service_instance: Optional[ProcurementService] = None


def get_service() -> ProcurementService:
    """
    Returns the singleton ProcurementService instance.
    Initializes lazily if not already warmed up in lifespan.
    """
    global _service_instance
    if _service_instance is None:
        dotenv_path = os.getenv("DOTENV_PATH", ".env")
        auto_build_env = env("AUTO_BUILD")
        auto_build = auto_build_env.lower() in ("true", "1", "yes") if auto_build_env is not None else None
        print(f"[api.dependencies] Initializing ProcurementService singleton (config: {dotenv_path})", file=sys.stderr)
        _service_instance = ProcurementService.get_instance(dotenv_path=dotenv_path, auto_build=auto_build)
        print("[api.dependencies] ProcurementService singleton ready.", file=sys.stderr)
    return _service_instance


def get_tenant_org_id(
    request: Request,
    x_organization_id: Optional[str] = Header(None, alias="X-Organization-Id")
) -> str:
    """
    Resolves the organization / tenant ID for multi-tenant isolation.
    Priority:
      1. Header 'X-Organization-Id'
      2. Environment variable 'DEFAULT_ORG_ID' (defaults to 'DGA')
    """
    if x_organization_id and x_organization_id.strip():
        return x_organization_id.strip()
    return os.getenv("DEFAULT_ORG_ID", "DGA").strip()
