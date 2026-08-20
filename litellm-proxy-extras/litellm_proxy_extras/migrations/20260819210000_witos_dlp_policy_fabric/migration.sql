-- WIT OS DLP Policy Federation (blueprint v2 §2.5).
-- Additive only: no existing table is altered.

CREATE TABLE IF NOT EXISTS "WITOS_DLPConnection" (
    "connection_id" TEXT NOT NULL,
    "provider" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "base_url" TEXT NOT NULL,
    "region" TEXT,
    "tenant_external_id" TEXT,
    "auth_type" TEXT NOT NULL,
    "secret_reference" TEXT NOT NULL,
    "capabilities_json" JSONB,
    "sync_mode" TEXT NOT NULL DEFAULT 'manual_approve',
    "sync_interval_min" INTEGER NOT NULL DEFAULT 60,
    "fail_mode" TEXT NOT NULL DEFAULT 'fail_open',
    "timeout_ms" INTEGER NOT NULL DEFAULT 800,
    "privacy_json" JSONB,
    "enabled" BOOLEAN NOT NULL DEFAULT true,
    "status" TEXT NOT NULL DEFAULT 'active',
    "last_sync_at" TIMESTAMP(3),
    "last_success_at" TIMESTAMP(3),
    "last_error" TEXT,
    "organization_id" TEXT,
    "created_by" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "WITOS_DLPConnection_pkey" PRIMARY KEY ("connection_id")
);

CREATE INDEX IF NOT EXISTS "WITOS_DLPConnection_provider_idx" ON "WITOS_DLPConnection"("provider");
CREATE INDEX IF NOT EXISTS "WITOS_DLPConnection_organization_id_idx" ON "WITOS_DLPConnection"("organization_id");

CREATE TABLE IF NOT EXISTS "WITOS_DLPPolicy" (
    "policy_id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "source" TEXT NOT NULL,
    "connection_id" TEXT,
    "external_policy_id" TEXT,
    "external_policy_version" TEXT,
    "mode" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'imported',
    "priority" INTEGER NOT NULL DEFAULT 100,
    "direction" TEXT NOT NULL DEFAULT 'both',
    "canonical_policy_json" JSONB NOT NULL,
    "external_policy_hash" TEXT,
    "canonical_policy_hash" TEXT NOT NULL,
    "compile_status" TEXT,
    "compile_error" TEXT,
    "approved_by" TEXT,
    "activated_at" TIMESTAMP(3),
    "last_synced_at" TIMESTAMP(3),
    "organization_id" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "WITOS_DLPPolicy_pkey" PRIMARY KEY ("policy_id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "WITOS_DLPPolicy_connection_id_external_policy_id_key" ON "WITOS_DLPPolicy"("connection_id", "external_policy_id");
CREATE INDEX IF NOT EXISTS "WITOS_DLPPolicy_status_idx" ON "WITOS_DLPPolicy"("status");
CREATE INDEX IF NOT EXISTS "WITOS_DLPPolicy_organization_id_status_idx" ON "WITOS_DLPPolicy"("organization_id", "status");

CREATE TABLE IF NOT EXISTS "WITOS_DLPPolicyVersion" (
    "policy_version_id" TEXT NOT NULL,
    "policy_id" TEXT NOT NULL,
    "version" INTEGER NOT NULL,
    "source_version" TEXT,
    "canonical_json" JSONB NOT NULL,
    "external_json_sanitized" JSONB,
    "hash" TEXT NOT NULL,
    "change_note" TEXT,
    "created_by" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "WITOS_DLPPolicyVersion_pkey" PRIMARY KEY ("policy_version_id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "WITOS_DLPPolicyVersion_policy_id_version_key" ON "WITOS_DLPPolicyVersion"("policy_id", "version");
CREATE INDEX IF NOT EXISTS "WITOS_DLPPolicyVersion_policy_id_created_at_idx" ON "WITOS_DLPPolicyVersion"("policy_id", "created_at");

CREATE TABLE IF NOT EXISTS "WITOS_DLPSyncRun" (
    "id" TEXT NOT NULL,
    "connection_id" TEXT NOT NULL,
    "started_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "finished_at" TIMESTAMP(3),
    "status" TEXT NOT NULL,
    "stats" JSONB NOT NULL,
    "error_detail" JSONB,

    CONSTRAINT "WITOS_DLPSyncRun_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "WITOS_DLPSyncRun_connection_id_started_at_idx" ON "WITOS_DLPSyncRun"("connection_id", "started_at");

CREATE TABLE IF NOT EXISTS "WITOS_DLPDecision" (
    "decision_id" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "request_id" TEXT NOT NULL,
    "scope_json" JSONB NOT NULL,
    "policy_id" TEXT NOT NULL,
    "policy_version" INTEGER NOT NULL,
    "provider" TEXT,
    "external_policy_id" TEXT,
    "direction" TEXT NOT NULL,
    "decision" TEXT NOT NULL,
    "risk_score" DECIMAL(5,2),
    "matched_classifiers" JSONB NOT NULL,
    "matched_rule_ids" JSONB,
    "vendor_request_id" TEXT,
    "evaluation_latency_ms" INTEGER,
    "fail_mode_triggered" BOOLEAN NOT NULL DEFAULT false,
    "redaction_performed" BOOLEAN NOT NULL DEFAULT false,
    "shadow" BOOLEAN NOT NULL DEFAULT false,
    "streaming_mode" TEXT,
    "prevented" BOOLEAN NOT NULL DEFAULT false,
    "organization_id" TEXT,
    "match_hash" TEXT,

    CONSTRAINT "WITOS_DLPDecision_pkey" PRIMARY KEY ("decision_id")
);

CREATE INDEX IF NOT EXISTS "WITOS_DLPDecision_policy_id_created_at_idx" ON "WITOS_DLPDecision"("policy_id", "created_at");
CREATE INDEX IF NOT EXISTS "WITOS_DLPDecision_request_id_idx" ON "WITOS_DLPDecision"("request_id");
CREATE INDEX IF NOT EXISTS "WITOS_DLPDecision_organization_id_created_at_idx" ON "WITOS_DLPDecision"("organization_id", "created_at");
CREATE INDEX IF NOT EXISTS "WITOS_DLPDecision_shadow_created_at_idx" ON "WITOS_DLPDecision"("shadow", "created_at");
