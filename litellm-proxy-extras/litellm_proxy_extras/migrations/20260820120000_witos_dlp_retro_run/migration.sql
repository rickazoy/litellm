-- WIT OS DLP retroactive policy simulation (blueprint v2 §2.7).
-- Additive only: no existing table is altered. Every column is a count, a
-- window bound, a hash or an id; no finding content is stored here.

CREATE TABLE IF NOT EXISTS "WITOS_DLPRetroRun" (
    "run_id" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "organization_id" TEXT,
    "name" TEXT,
    "candidate_policy_hash" TEXT NOT NULL,
    "candidate_policy_name" TEXT NOT NULL,
    "window_start" TIMESTAMP(3) NOT NULL,
    "window_end" TIMESTAMP(3) NOT NULL,
    "filters_json" JSONB NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'complete',
    "scanned_decisions" INTEGER NOT NULL,
    "evaluated_requests" INTEGER NOT NULL,
    "matched_requests" INTEGER NOT NULL,
    "indeterminate_requests" INTEGER NOT NULL,
    "newly_blocked" INTEGER NOT NULL,
    "newly_restricted" INTEGER NOT NULL,
    "newly_allowed" INTEGER NOT NULL,
    "unchanged" INTEGER NOT NULL,
    "aggregates_json" JSONB NOT NULL,
    "indeterminate_json" JSONB NOT NULL,
    "sample_decision_ids" JSONB NOT NULL,
    "row_cap" INTEGER NOT NULL,
    "truncated" BOOLEAN NOT NULL DEFAULT false,

    CONSTRAINT "WITOS_DLPRetroRun_pkey" PRIMARY KEY ("run_id")
);

CREATE INDEX IF NOT EXISTS "WITOS_DLPRetroRun_organization_id_created_at_idx" ON "WITOS_DLPRetroRun"("organization_id", "created_at");
CREATE INDEX IF NOT EXISTS "WITOS_DLPRetroRun_candidate_policy_hash_idx" ON "WITOS_DLPRetroRun"("candidate_policy_hash");
