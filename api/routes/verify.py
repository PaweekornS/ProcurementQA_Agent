# -*- coding: utf-8 -*-
"""
api/routes/verify.py

Statutory compliance verification endpoint for threshold checking and method validation.
"""

from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, status
from api.dependencies import get_service
from api.schemas import VerifyComplianceRequest
from core.service import ProcurementService

router = APIRouter(prefix="/api/v1", tags=["Statutory Compliance Verification"])


@router.post(
    "/verify",
    summary="Verify Procurement Method Compliance",
    status_code=status.HTTP_200_OK
)
def verify_procurement_compliance(
    payload: VerifyComplianceRequest,
    service: ProcurementService = Depends(get_service)
) -> Dict[str, Any]:
    """
    Evaluates whether a proposed procurement method satisfies statutory monetary thresholds
    and legal prerequisites under the Thai Public Procurement Act B.E. 2560.
    """
    if not payload.procurement_item.strip() or not payload.proposed_method.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="'procurement_item' and 'proposed_method' are required fields."
        )

    try:
        return service.verify_compliance(
            procurement_item=payload.procurement_item,
            estimated_budget=payload.estimated_budget,
            proposed_method=payload.proposed_method,
            justification_reason=payload.justification_reason
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Compliance verification failed: {type(exc).__name__}: {str(exc)}"
        )
