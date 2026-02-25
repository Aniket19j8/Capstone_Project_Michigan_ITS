from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


UNKNOWN = "unknown"


class TicketSource(str, Enum):
    VOICE = "voice"
    WEB = "web"
    API = "api"
    INTEGRATION = "integration"


class TicketStatus(str, Enum):
    NEW = "new"
    TRIAGED = "triaged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class Intent(str, Enum):
    BUG = "bug"
    FEATURE = "feature"
    INCIDENT = "incident"
    QUESTION = "question"


class Frequency(str, Enum):
    ALWAYS = "always"
    SOMETIMES = "sometimes"
    ONCE = "once"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"


class Priority(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class WhisperMeta(BaseModel):
    model: str
    device: str
    compute_type: str
    language: Optional[str] = None
    duration: Optional[float] = None


class Transcript(BaseModel):
    transcript: str
    segments: list[TranscriptSegment] = Field(default_factory=list)
    meta: WhisperMeta


class Content(BaseModel):
    intent: Intent = Intent.BUG
    title: str = UNKNOWN
    summary: str = UNKNOWN
    description: Optional[str] = None
    steps_to_reproduce: list[str] = Field(default_factory=list)
    expected_behavior: str = UNKNOWN
    observed_behavior: str = UNKNOWN
    frequency: Frequency = Frequency.UNKNOWN
    workaround: str = UNKNOWN


class EnvironmentOS(BaseModel):
    name: str = UNKNOWN
    version: str = UNKNOWN


class EnvironmentDevice(BaseModel):
    manufacturer: str = UNKNOWN
    model: str = UNKNOWN


class EnvironmentApp(BaseModel):
    name: str = UNKNOWN
    version: str = UNKNOWN
    build: str = UNKNOWN


class Network(BaseModel):
    type: str = UNKNOWN  # wifi/cellular/ethernet/unknown


class Environment(BaseModel):
    os: EnvironmentOS = Field(default_factory=EnvironmentOS)
    device: EnvironmentDevice = Field(default_factory=EnvironmentDevice)
    app: EnvironmentApp = Field(default_factory=EnvironmentApp)
    browser: Optional[dict[str, Any]] = None
    region: str = UNKNOWN
    network: Network = Field(default_factory=Network)
    feature_flags: list[str] = Field(default_factory=list)


class ErrorSignature(BaseModel):
    type: str = "unknown"  # exception/error_code/log_line/stack_trace/endpoint
    value: str = UNKNOWN
    confidence: Optional[float] = None


class Evidence(BaseModel):
    error_signatures: list[ErrorSignature] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)


class UsersAffected(BaseModel):
    estimate: Optional[int] = None
    confidence: Optional[float] = None


class Impact(BaseModel):
    customer_facing: Optional[bool] = None
    business_impact: str = UNKNOWN
    urgency_note: str = UNKNOWN
    users_affected: UsersAffected = Field(default_factory=UsersAffected)


class ClassificationConfidence(BaseModel):
    intent: Optional[float] = None
    component: Optional[float] = None
    severity: Optional[float] = None
    priority: Optional[float] = None


class Classification(BaseModel):
    component: str = UNKNOWN
    tags: list[str] = Field(default_factory=list)
    severity: Optional[Severity] = None
    priority: Optional[Priority] = None
    confidence: ClassificationConfidence = Field(default_factory=ClassificationConfidence)
    reason_codes: list[str] = Field(default_factory=list)


class Routing(BaseModel):
    team: str = UNKNOWN
    assignee: Optional[str] = None
    watchers: list[str] = Field(default_factory=list)


class Quality(BaseModel):
    missing_fields: list[str] = Field(default_factory=list)
    followup_questions: list[str] = Field(default_factory=list)
    completeness_score: Optional[float] = None


class IntakeArtifacts(BaseModel):
    transcript: str
    transcript_segments: list[TranscriptSegment] = Field(default_factory=list)
    whisper: WhisperMeta


class ExtractionOutput(BaseModel):
    """
    This is the ONLY thing the LLM needs to produce.
    We'll add ids/timestamps ourselves.
    """
    content: Content
    environment: Environment = Field(default_factory=Environment)
    evidence: Evidence = Field(default_factory=Evidence)
    impact: Impact = Field(default_factory=Impact)
    classification: Classification = Field(default_factory=Classification)
    routing: Routing = Field(default_factory=Routing)
    quality: Quality = Field(default_factory=Quality)


class TicketDraft(BaseModel):
    schema_version: str = "1.0"
    ticket_id: UUID
    status: TicketStatus = TicketStatus.NEW
    source: TicketSource = TicketSource.VOICE
    created_at: datetime
    updated_at: Optional[datetime] = None

    locale: Optional[str] = None
    timezone: Optional[str] = None

    content: Content
    environment: Environment
    evidence: Evidence
    impact: Impact
    classification: Classification
    routing: Routing

    intake_artifacts: IntakeArtifacts
    quality: Quality