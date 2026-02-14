"""Pydantic response models for the Verifily SDK.

These mirror the server-side schemas so callers get typed access to fields.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Health / Ready ───────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    time: str
    mode: str


class ReadyResponse(BaseModel):
    status: str
    checks: Dict[str, str]
    error: Optional[str] = None


# ── Pipeline ─────────────────────────────────────────────────────

class ContaminationSummary(BaseModel):
    status: str
    exact_overlaps: int
    near_duplicates: int
    exact_overlap_fraction: float
    near_duplicate_fraction: float


class ContractSummary(BaseModel):
    valid: bool
    checks: List[Dict[str, str]]
    has_eval: bool


class PipelineResponse(BaseModel):
    exit_code: int
    decision: Dict[str, Any]
    contamination: Optional[ContaminationSummary] = None
    contract: Optional[ContractSummary] = None
    report_summary: Optional[Dict[str, Any]] = None
    output_dir: Optional[str] = None
    elapsed_ms: int
    config_path: str
    usage: Optional[Dict[str, Any]] = None


# ── Contamination ────────────────────────────────────────────────

class ContaminationResponse(BaseModel):
    status: str
    exit_code: int
    train_rows: int
    eval_rows: int
    exact_overlaps: int
    exact_overlap_fraction: float
    near_duplicates: int
    near_duplicate_fraction: float
    reasons: List[str]
    elapsed_ms: int


# ── Report ───────────────────────────────────────────────────────

class ReportResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    row_count: int
    schema_type: str = Field(alias="schema")
    field_stats: Dict[str, Any]
    tag_distribution: Dict[str, Any]
    pii_summary: Dict[str, int]
    pii_clean: bool
    sample_rows: Optional[List[Dict[str, Any]]] = None
    exit_code: int = 0
    elapsed_ms: int


# ── Usage Accounting ────────────────────────────────────────────

class UsageBucket(BaseModel):
    api_key_id: Optional[str] = None
    project_id: Optional[str] = None
    requests: int = 0
    decisions_ship: int = 0
    decisions_dont_ship: int = 0
    decisions_investigate: int = 0
    rows_in: int = 0
    rows_out: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    elapsed_ms_sum: int = 0


class UsageResponse(BaseModel):
    buckets: Optional[List[UsageBucket]] = None
    total: Optional[Dict[str, int]] = None


# ── Async Jobs ─────────────────────────────────────────────────

class JobSubmitResponse(BaseModel):
    job_id: str
    status: str
    request_id: Optional[str] = None
    project_id: str = "default"


class JobMetaResponse(BaseModel):
    job_id: str
    type: str
    status: str
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    request_id: Optional[str] = None
    api_key_id: str = "anonymous"
    project_id: str = "default"
    input_hash: Optional[str] = None
    result_path: Optional[str] = None
    error: Optional[str] = None


class JobListResponse(BaseModel):
    jobs: List[JobMetaResponse]


# ── Monitor ─────────────────────────────────────────────────

# ── Retrain ──────────────────────────────────────────────────

class RetrainResponse(BaseModel):
    job_type: str = "RETRAIN"
    run_dir: str
    decision: Dict[str, Any]
    exit_code: int
    eval_summary: Dict[str, Any]
    usage: Optional[Dict[str, Any]] = None
    artifacts: Dict[str, str]
    elapsed_ms: int


# ── Monitor ─────────────────────────────────────────────────

class MonitorStartResponse(BaseModel):
    monitor_id: str
    status: str


class MonitorStatusResponse(BaseModel):
    monitor_id: str
    status: str
    tick_count: int
    last_tick: Optional[Dict[str, Any]] = None
    config: Dict[str, Any]
    error: Optional[str] = None


class MonitorHistoryResponse(BaseModel):
    monitor_id: str
    ticks: List[Dict[str, Any]]
    total_ticks: int


# ── Orgs / Projects ─────────────────────────────────────────────

class OrgResponse(BaseModel):
    id: str
    name: str
    created_at: float
    created_by: str


class OrgListResponse(BaseModel):
    orgs: List[OrgResponse]


class ProjectResponse(BaseModel):
    id: str
    org_id: str
    name: str
    created_at: float
    created_by: str


class ProjectListResponse(BaseModel):
    projects: List[ProjectResponse]


# ── Admin (Enterprise Trust) ───────────────────────────────────

class AdminProjectResponse(BaseModel):
    id: str
    name: str
    created_at: float


class AdminProjectListResponse(BaseModel):
    projects: List[AdminProjectResponse]


class AdminKeyResponse(BaseModel):
    id: str
    name: str
    created_at: float
    scopes: List[str]
    projects_allowed: List[str]
    disabled: bool


class AdminKeyListResponse(BaseModel):
    keys: List[AdminKeyResponse]


# ── Billing ────────────────────────────────────────────────────

class BillingEventResponse(BaseModel):
    ts: float
    api_key_id: str = "anonymous"
    project_id: str = "default"
    endpoint: str = ""
    units: Dict[str, int] = Field(default_factory=dict)
    request_id: Optional[str] = None
    job_id: Optional[str] = None
    status_code: int = 200


class BillingEventsResponse(BaseModel):
    events: List[BillingEventResponse]
    total: int


class InvoiceLineResponse(BaseModel):
    label: str
    unit_type: str
    quantity: int
    included: int
    overage: int
    unit_price_cents: int
    amount_cents: int


class InvoicePreviewResponse(BaseModel):
    project_id: str
    plan_id: str
    window_start: float
    window_end: float
    lines: List[InvoiceLineResponse]
    monthly_base_cents: int
    subtotal_cents: int
    tax_cents: int = 0
    total_cents: int


# ── Billing (extended) ─────────────────────────────────────────

class BillingPlanResponse(BaseModel):
    id: str
    name: str
    monthly_base_cents: int
    included_requests: int
    included_rows: int
    included_bytes: int
    price_per_1k_rows: int
    price_per_request: int
    price_per_mb: int
    price_per_decision: int


class BillingPlansResponse(BaseModel):
    plans: List[BillingPlanResponse]


class BillingEstimateResponse(BaseModel):
    plan_id: str
    window_start: float
    window_end: float
    project_id: Optional[str] = None
    lines: List[InvoiceLineResponse]
    monthly_base_cents: int
    subtotal_cents: int
    tax_cents: int = 0
    total_cents: int


class BillingInvoiceResponse(BaseModel):
    invoice_id: str
    plan_id: str
    project_id: str
    period_start: str
    period_end: str
    lines: List[InvoiceLineResponse]
    monthly_base_cents: int
    subtotal_cents: int
    tax_cents: int = 0
    total_cents: int
    invoice_path: Optional[str] = None


# ── Teams (RBAC) ──────────────────────────────────────────────

class TeamsWhoamiResponse(BaseModel):
    api_key_id: str
    org_id: str
    scopes: List[str]
    project_ids: List[str]


class TeamsCreateOrgResponse(BaseModel):
    id: str
    name: str
    created_at: float


class TeamsCreateUserResponse(BaseModel):
    id: str
    email: str
    name: str
    created_at: float


class TeamsAddMembershipResponse(BaseModel):
    user_id: str
    org_id: str
    role: str
    created_at: float


class TeamsCreateProjectResponse(BaseModel):
    id: str
    org_id: str
    name: str
    created_at: float


class TeamsProjectListResponse(BaseModel):
    projects: List[TeamsCreateProjectResponse]


class TeamsIssueApiKeyResponse(BaseModel):
    id: str
    org_id: str
    name: str
    scopes: List[str]
    project_ids: List[str]
    created_at: float
    created_by: str


# ── Effective Config ───────────────────────────────────────────

class EffectiveConfigResponse(BaseModel):
    config: Dict[str, Any]


# ── Risk Score / Health Index ─────────────────────────────────

class ScoreComponentResponse(BaseModel):
    name: str
    value: float
    weight: float
    detail: str
    contribution: float


class RiskScoreResponse(BaseModel):
    total: float
    level: str
    summary: str
    components: List[ScoreComponentResponse]


class HealthIndexResponse(BaseModel):
    total: float
    level: str
    summary: str
    components: List[ScoreComponentResponse]


class ScoreResponse(BaseModel):
    risk_score: RiskScoreResponse
    health_index: HealthIndexResponse
    verdict: str
    recommendations: List[str]


# ── Model Registry ────────────────────────────────────────────

class ModelRecordResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    version: str
    created_at: str
    source_run: str
    decision: str
    risk_score: float
    health_index: float
    metrics: Dict[str, float] = Field(default_factory=dict)
    lineage_hash: str = ""
    stage: str = "none"
    registered_by: str = "unknown"
    promotion_history: List[Dict[str, Any]] = Field(default_factory=list)


class RegisterModelResponse(BaseModel):
    success: bool
    record: Optional[ModelRecordResponse] = None
    error: Optional[str] = None


class PromoteModelResponse(BaseModel):
    success: bool
    record: Optional[ModelRecordResponse] = None
    error: Optional[str] = None


class ListModelsResponse(BaseModel):
    models: List[ModelRecordResponse]
    total: int


class ModelHistoryResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    versions: List[ModelRecordResponse]


# ── Lineage Graph ──────────────────────────────────────────────

class LineageNodeResponse(BaseModel):
    id: str
    type: str
    label: str
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LineageEdgeResponse(BaseModel):
    source: str
    target: str
    relation: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LineageResponse(BaseModel):
    version: str
    root_id: str
    created_at: str
    nodes: List[LineageNodeResponse]
    edges: List[LineageEdgeResponse]


# ── Budget ──────────────────────────────────────────────────────

class BudgetPolicyResponse(BaseModel):
    daily_limit_units: int
    monthly_limit_units: int
    hard_block: bool
    reset_hour_utc: int


class BudgetUsageResponse(BaseModel):
    mode: str  # pass, warn, block
    daily_used: int
    daily_remaining: int
    daily_limit: int
    monthly_used: int
    monthly_remaining: int
    monthly_limit: int


class BudgetStatusResponse(BaseModel):
    project_id: str
    policy: BudgetPolicyResponse
    usage: BudgetUsageResponse
    next_reset: str
    seconds_until_reset: int


# ── Stripe / Subscriptions ─────────────────────────────────────

class CheckoutResponse(BaseModel):
    checkout_url: str
    stripe_customer_id: str
    plan: str


class SubscriptionResponse(BaseModel):
    org_id: str
    project_id: str
    stripe_customer_id: str = ""
    stripe_subscription_id: str = ""
    plan: str = "free"
    status: str = "incomplete"
    created_at: float = 0.0
    updated_at: float = 0.0
