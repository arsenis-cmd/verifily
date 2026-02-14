"""Pydantic request/response models for the Verifily local API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Health ───────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    time: str
    mode: str = "local"


# ── Ready ────────────────────────────────────────────────────────

class ReadyResponse(BaseModel):
    status: str
    checks: Dict[str, str]
    error: Optional[str] = None


# ── Pipeline ─────────────────────────────────────────────────────

class PipelineRequest(BaseModel):
    config_path: Optional[str] = Field(
        None,
        description="Absolute path to a verifily.yaml pipeline config file.",
    )
    project_path: Optional[str] = Field(
        None,
        description="Path to project directory containing verifily.yaml.",
    )
    plan: bool = Field(
        False,
        description="If true, return what would happen without writing outputs.",
    )
    ci: bool = Field(
        True,
        description="Run in CI mode (structured JSON decision, clean exit codes).",
    )
    overrides: Optional[Dict[str, Any]] = Field(
        None,
        description="Key-value overrides to merge into the pipeline config.",
    )
    project_id: Optional[str] = Field(
        None,
        description="Project identifier for usage accounting.",
    )


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
    risk_score: Optional[RiskScoreResponse] = None
    health_index: Optional[HealthIndexResponse] = None


# ── Contamination ────────────────────────────────────────────────

class ContaminationRequest(BaseModel):
    train_path: str
    eval_path: str
    jaccard_cutoff: float = Field(0.70, ge=0.0, le=1.0)
    no_write: bool = Field(True, description="If true, do not write results to disk.")
    out_path: Optional[str] = Field(None, description="Path to write JSON results.")
    project_id: Optional[str] = Field(
        None,
        description="Project identifier for usage accounting.",
    )


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

class ReportRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    dataset_path: str
    schema_type: str = Field("sft", alias="schema", description="Dataset schema: sft or classification.")
    sample: int = Field(0, ge=0, description="Number of sample rows to include (0=none).")
    out_dir: Optional[str] = Field(None, description="Directory to write report JSON.")
    project_id: Optional[str] = Field(
        None,
        description="Project identifier for usage accounting.",
    )


class ReportResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

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


# ── Classify ─────────────────────────────────────────────────

# ── Monitor ─────────────────────────────────────────────────

class MonitorStartRequest(BaseModel):
    config_path: str = Field(..., description="Path to verifily.yaml pipeline config.")
    interval_seconds: int = Field(60, ge=1, description="Seconds between ticks.")
    max_ticks: int = Field(0, ge=0, description="Max ticks (0=unlimited).")
    rolling_window: int = Field(20, ge=1, description="Max ticks kept in history.")
    project_id: Optional[str] = Field(None, description="Project identifier.")


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


# ── Retrain ──────────────────────────────────────────────────

class RetrainRequest(BaseModel):
    project_id: Optional[str] = Field(None, description="Project identifier.")
    dataset_dir: str = Field(..., description="Path to dataset artifact directory (must contain dataset.jsonl + hashes.json).")
    base_run_dir: Optional[str] = Field(None, description="Baseline run to compare against.")
    contaminated_run_dir: Optional[str] = Field(None, description="Run with contamination to test degradation.")
    metric: str = Field("f1", description="Primary metric for evaluation.")
    mode: str = Field("mock", description="Training mode: 'mock' (default, safe) or 'real' (requires env var).")
    output_dir: Optional[str] = Field(None, description="Output directory for run artifacts.")
    seed: int = Field(42, description="Random seed for deterministic mock metrics.")
    notes: Optional[str] = Field(None, description="Optional notes for audit log.")


class RetrainResponse(BaseModel):
    job_type: str = "RETRAIN"
    run_dir: str
    decision: Dict[str, Any]
    exit_code: int
    eval_summary: Dict[str, Any]
    usage: Optional[Dict[str, Any]] = None
    artifacts: Dict[str, str]
    elapsed_ms: int


# ── Classify ─────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    dataset_path: Optional[str] = Field(
        None,
        description="Path to dataset file (JSONL or CSV).",
    )
    output_dir: Optional[str] = Field(
        None,
        description="Directory to write classification artifacts.",
    )
    max_rows_scan: int = Field(
        500,
        ge=0,
        description="Max rows to scan for heuristics (0=all).",
    )
    export_buckets: bool = Field(
        False,
        description="If true and output_dir is set, write per-bucket JSONL files and suggested_next_steps.json.",
    )
    min_bucket_rows: int = Field(
        1,
        ge=1,
        description="Minimum rows for a bucket to be exported (only used when export_buckets=true).",
    )
    project_id: Optional[str] = Field(
        None,
        description="Project identifier for usage accounting.",
    )


# ── Orgs / Projects ─────────────────────────────────────────────

class CreateOrgRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Organization name.")


class OrgResponse(BaseModel):
    id: str
    name: str
    created_at: float
    created_by: str


class OrgListResponse(BaseModel):
    orgs: List[OrgResponse]


class CreateProjectRequest(BaseModel):
    org_id: str = Field(..., description="Organization ID this project belongs to.")
    name: str = Field(..., min_length=1, max_length=100, description="Project name.")


class ProjectResponse(BaseModel):
    id: str
    org_id: str
    name: str
    created_at: float
    created_by: str


class ProjectListResponse(BaseModel):
    projects: List[ProjectResponse]


class AddMembershipRequest(BaseModel):
    api_key_id: str = Field(..., description="API key ID to add as member.")
    role: str = Field("MEMBER", description="Role: OWNER, ADMIN, or MEMBER.")


class MembershipResponse(BaseModel):
    org_id: str
    api_key_id: str
    role: str
    created_at: float


class MembershipListResponse(BaseModel):
    memberships: List[MembershipResponse]


# ── Admin (Enterprise Trust) ───────────────────────────────────

class AdminCreateProjectRequest(BaseModel):
    id: str = Field(..., min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=100)


class AdminProjectResponse(BaseModel):
    id: str
    name: str
    created_at: float


class AdminProjectListResponse(BaseModel):
    projects: List[AdminProjectResponse]


class AdminCreateKeyRequest(BaseModel):
    id: str = Field(..., min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=100)
    raw_key: str = Field(..., min_length=8)
    scopes: List[str]
    projects_allowed: List[str]


class AdminKeyResponse(BaseModel):
    id: str
    name: str
    created_at: float
    scopes: List[str]
    projects_allowed: List[str]
    disabled: bool


class AdminKeyListResponse(BaseModel):
    keys: List[AdminKeyResponse]


class AdminRotateKeyRequest(BaseModel):
    raw_key: str = Field(..., min_length=8)


# ── Billing ─────────────────────────────────────────────────────

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


class BillingInvoiceRequest(BaseModel):
    plan: str = Field(..., description="Plan ID: FREE, STARTER, PRO, ENTERPRISE.")
    period_start: str = Field(..., description="ISO datetime for period start.")
    period_end: str = Field(..., description="ISO datetime for period end.")
    project_id: Optional[str] = Field(None, description="Filter to specific project.")


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


class BillingUsageExportResponse(BaseModel):
    format: str
    rows: int
    content: Optional[str] = None
    path: Optional[str] = None


# ── Teams (RBAC) ──────────────────────────────────────────────

class TeamsCreateOrgRequest(BaseModel):
    id: str = Field(..., min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=100)


class TeamsCreateOrgResponse(BaseModel):
    id: str
    name: str
    created_at: float


class TeamsCreateUserRequest(BaseModel):
    id: str = Field(..., min_length=1, max_length=64)
    email: str = Field(..., min_length=1, max_length=200)
    name: str = Field(..., min_length=1, max_length=100)


class TeamsCreateUserResponse(BaseModel):
    id: str
    email: str
    name: str
    created_at: float


class TeamsAddMembershipRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    org_id: str = Field(..., min_length=1)
    role: str = Field("member", description="Role: owner, admin, member, or readonly.")


class TeamsAddMembershipResponse(BaseModel):
    user_id: str
    org_id: str
    role: str
    created_at: float


class TeamsCreateProjectRequest(BaseModel):
    id: str = Field(..., min_length=1, max_length=64)
    org_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1, max_length=100)


class TeamsCreateProjectResponse(BaseModel):
    id: str
    org_id: str
    name: str
    created_at: float


class TeamsProjectListResponse(BaseModel):
    projects: List[TeamsCreateProjectResponse]


class TeamsIssueApiKeyRequest(BaseModel):
    id: str = Field(..., min_length=1, max_length=64)
    org_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1, max_length=100)
    raw_key: str = Field(..., min_length=8)
    scopes: List[str]
    project_ids: List[str]
    created_by: str = Field(..., min_length=1)


class TeamsIssueApiKeyResponse(BaseModel):
    id: str
    org_id: str
    name: str
    scopes: List[str]
    project_ids: List[str]
    created_at: float
    created_by: str


class TeamsWhoamiResponse(BaseModel):
    api_key_id: str
    org_id: str
    scopes: List[str]
    project_ids: List[str]


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


# ── Model Registry ─────────────────────────────────────────────

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


class RegisterModelRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    run_dir: str
    model_id: Optional[str] = None
    version: Optional[str] = None


class RegisterModelResponse(BaseModel):
    success: bool
    record: Optional[ModelRecordResponse] = None
    error: Optional[str] = None


class PromoteModelRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    version: str
    target_stage: str
    reason: str = ""


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


# ── Org Mode (Multi-tenant) ───────────────────────────────────────

class CreateOrgRequest(BaseModel):
    name: str


class CreateOrgResponse(BaseModel):
    org_id: str
    name: str
    created_at: str


class CreateProjectRequest(BaseModel):
    org_id: str
    name: str


class CreateProjectResponse(BaseModel):
    project_id: str
    org_id: str
    name: str
    created_at: str


class CreateKeyRequest(BaseModel):
    project_id: str
    role: str  # admin, dev, viewer
    label: Optional[str] = None


class CreateKeyResponse(BaseModel):
    secret: str  # Returned ONCE
    key_id: str
    org_id: str
    project_id: str
    role: str
    created_at: str
    label: Optional[str] = None


class RevokeKeyResponse(BaseModel):
    key_id: str
    revoked_at: str
    was_active: bool


class ListProjectsResponse(BaseModel):
    projects: List[CreateProjectResponse]


class ListKeysResponse(BaseModel):
    keys: List[Dict[str, Any]]


class KeyInfoResponse(BaseModel):
    key_id: str
    org_id: str
    project_id: str
    role: str
    created_at: str
    revoked_at: Optional[str] = None
    label: Optional[str] = None
    is_active: bool


# ── Stripe / Subscriptions ─────────────────────────────────────

class CheckoutRequest(BaseModel):
    plan: str = Field("pro", description="Plan to subscribe to (pro for now).")
    project_id: Optional[str] = Field(None, description="Project ID.")
    org_id: Optional[str] = Field(None, description="Organization ID.")


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
