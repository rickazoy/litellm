-- CreateTable
CREATE TABLE IF NOT EXISTS "WITOS_FinOpsUsageHourly" (
    "id" TEXT NOT NULL,
    "bucket_start_utc" TIMESTAMP(3) NOT NULL,
    "scope_type" TEXT NOT NULL,
    "scope_id" TEXT NOT NULL,
    "scope_label" TEXT,
    "organization_id" TEXT,
    "team_id" TEXT,
    "model" TEXT,
    "model_group" TEXT,
    "provider" TEXT,
    "call_type" TEXT,
    "request_count" INTEGER NOT NULL DEFAULT 0,
    "successful_requests" INTEGER NOT NULL DEFAULT 0,
    "failed_requests" INTEGER NOT NULL DEFAULT 0,
    "prompt_tokens" BIGINT NOT NULL DEFAULT 0,
    "completion_tokens" BIGINT NOT NULL DEFAULT 0,
    "total_tokens" BIGINT NOT NULL DEFAULT 0,
    "cache_read_tokens" BIGINT NOT NULL DEFAULT 0,
    "cache_creation_tokens" BIGINT NOT NULL DEFAULT 0,
    "spend_usd" DECIMAL(18,8) NOT NULL DEFAULT 0,
    "avg_latency_ms" INTEGER,
    "p95_latency_ms" INTEGER,
    "breakdown" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "WITOS_FinOpsUsageHourly_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE IF NOT EXISTS "WITOS_FinOpsUsageDaily" (
    "id" TEXT NOT NULL,
    "bucket_date" TIMESTAMP(3) NOT NULL,
    "scope_type" TEXT NOT NULL,
    "scope_id" TEXT NOT NULL,
    "scope_label" TEXT,
    "organization_id" TEXT,
    "team_id" TEXT,
    "model_group" TEXT,
    "provider" TEXT,
    "request_count" INTEGER NOT NULL DEFAULT 0,
    "successful_requests" INTEGER NOT NULL DEFAULT 0,
    "failed_requests" INTEGER NOT NULL DEFAULT 0,
    "prompt_tokens" BIGINT NOT NULL DEFAULT 0,
    "completion_tokens" BIGINT NOT NULL DEFAULT 0,
    "total_tokens" BIGINT NOT NULL DEFAULT 0,
    "cache_read_tokens" BIGINT NOT NULL DEFAULT 0,
    "cache_creation_tokens" BIGINT NOT NULL DEFAULT 0,
    "spend_usd" DECIMAL(18,8) NOT NULL DEFAULT 0,
    "breakdown" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "WITOS_FinOpsUsageDaily_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE IF NOT EXISTS "WITOS_FinOpsQuota" (
    "quota_id" TEXT NOT NULL,
    "scope_type" TEXT NOT NULL,
    "scope_id" TEXT NOT NULL,
    "metric" TEXT NOT NULL,
    "limit_value" DECIMAL(24,4) NOT NULL,
    "period" TEXT NOT NULL,
    "period_start" TIMESTAMP(3) NOT NULL,
    "period_end" TIMESTAMP(3),
    "reset_at" TIMESTAMP(3),
    "source_type" TEXT NOT NULL,
    "source_id" TEXT,
    "soft_threshold_pct" DECIMAL(5,2),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "WITOS_FinOpsQuota_pkey" PRIMARY KEY ("quota_id")
);

-- CreateTable
CREATE TABLE IF NOT EXISTS "WITOS_FinOpsForecast" (
    "forecast_id" TEXT NOT NULL,
    "scope_type" TEXT NOT NULL,
    "scope_id" TEXT NOT NULL,
    "generated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "forecast_start" TIMESTAMP(3) NOT NULL,
    "forecast_end" TIMESTAMP(3) NOT NULL,
    "metric" TEXT NOT NULL,
    "strategy" TEXT NOT NULL,
    "algorithm" TEXT NOT NULL,
    "algorithm_version" TEXT NOT NULL,
    "history_days" INTEGER NOT NULL,
    "training_points" INTEGER NOT NULL,
    "trend_regime" TEXT NOT NULL,
    "wape" DECIMAL(8,4),
    "bias" DECIMAL(8,4),
    "quality_score" INTEGER,
    "pricing_snapshot_hash" TEXT,
    "forecast_json" JSONB NOT NULL,
    "proj_eom" DECIMAL(18,4),
    "proj_eoq" DECIMAL(18,4),
    "burn_rate_daily" DECIMAL(18,4),
    "is_latest" BOOLEAN NOT NULL DEFAULT true,

    CONSTRAINT "WITOS_FinOpsForecast_pkey" PRIMARY KEY ("forecast_id")
);

-- CreateTable
CREATE TABLE IF NOT EXISTS "WITOS_FinOpsRunway" (
    "id" TEXT NOT NULL,
    "computed_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "quota_id" TEXT NOT NULL,
    "scope_type" TEXT NOT NULL,
    "scope_id" TEXT NOT NULL,
    "consumed" DECIMAL(24,4) NOT NULL,
    "remaining" DECIMAL(24,4) NOT NULL,
    "exhaustion_p50" TIMESTAMP(3),
    "exhaustion_p90" TIMESTAMP(3),
    "prob_exhaust_before_reset" DECIMAL(5,4),
    "prob_overrun_this_period" DECIMAL(5,4),
    "survives_cycle" BOOLEAN NOT NULL,
    "is_latest" BOOLEAN NOT NULL DEFAULT true,

    CONSTRAINT "WITOS_FinOpsRunway_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE IF NOT EXISTS "WITOS_FinOpsScenario" (
    "scenario_id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "scope_type" TEXT NOT NULL,
    "scope_id" TEXT NOT NULL,
    "assumptions_json" JSONB NOT NULL,
    "created_by" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "WITOS_FinOpsScenario_pkey" PRIMARY KEY ("scenario_id")
);

-- CreateTable
CREATE TABLE IF NOT EXISTS "WITOS_FinOpsAlertRule" (
    "alert_id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "scope_type" TEXT NOT NULL,
    "scope_id" TEXT NOT NULL,
    "condition_type" TEXT NOT NULL,
    "threshold" DECIMAL(12,4) NOT NULL,
    "lookahead_days" INTEGER,
    "channels" JSONB NOT NULL,
    "cooldown_min" INTEGER NOT NULL DEFAULT 240,
    "enabled" BOOLEAN NOT NULL DEFAULT true,
    "last_triggered_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "WITOS_FinOpsAlertRule_pkey" PRIMARY KEY ("alert_id")
);

-- CreateTable
CREATE TABLE IF NOT EXISTS "WITOS_FinOpsAlertEvent" (
    "id" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "alert_id" TEXT,
    "alert_type" TEXT NOT NULL,
    "severity" TEXT NOT NULL,
    "scope_type" TEXT NOT NULL,
    "scope_id" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "body" JSONB NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'open',
    "acked_by" TEXT,
    "notified_via" JSONB,

    CONSTRAINT "WITOS_FinOpsAlertEvent_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE IF NOT EXISTS "WITOS_BackgroundJobLease" (
    "job_name" TEXT NOT NULL,
    "owner_id" TEXT NOT NULL,
    "lease_until" TIMESTAMP(3) NOT NULL,
    "heartbeat_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "WITOS_BackgroundJobLease_pkey" PRIMARY KEY ("job_name")
);

-- CreateTable
CREATE TABLE IF NOT EXISTS "WITOS_FinOpsConfig" (
    "config_key" TEXT NOT NULL,
    "value_json" JSONB NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "WITOS_FinOpsConfig_pkey" PRIMARY KEY ("config_key")
);

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "WITOS_FinOpsUsageHourly_bucket_start_utc_scope_type_scope_i_key" ON "WITOS_FinOpsUsageHourly"("bucket_start_utc", "scope_type", "scope_id");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "WITOS_FinOpsUsageHourly_scope_type_scope_id_bucket_start_ut_idx" ON "WITOS_FinOpsUsageHourly"("scope_type", "scope_id", "bucket_start_utc");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "WITOS_FinOpsUsageHourly_bucket_start_utc_idx" ON "WITOS_FinOpsUsageHourly"("bucket_start_utc");

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "WITOS_FinOpsUsageDaily_bucket_date_scope_type_scope_id_key" ON "WITOS_FinOpsUsageDaily"("bucket_date", "scope_type", "scope_id");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "WITOS_FinOpsUsageDaily_scope_type_scope_id_bucket_date_idx" ON "WITOS_FinOpsUsageDaily"("scope_type", "scope_id", "bucket_date");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "WITOS_FinOpsUsageDaily_bucket_date_idx" ON "WITOS_FinOpsUsageDaily"("bucket_date");

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "WITOS_FinOpsQuota_source_type_scope_type_scope_id_metric_pe_key" ON "WITOS_FinOpsQuota"("source_type", "scope_type", "scope_id", "metric", "period");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "WITOS_FinOpsQuota_scope_type_scope_id_idx" ON "WITOS_FinOpsQuota"("scope_type", "scope_id");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "WITOS_FinOpsForecast_scope_type_scope_id_metric_is_latest_idx" ON "WITOS_FinOpsForecast"("scope_type", "scope_id", "metric", "is_latest");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "WITOS_FinOpsForecast_generated_at_idx" ON "WITOS_FinOpsForecast"("generated_at");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "WITOS_FinOpsRunway_quota_id_is_latest_idx" ON "WITOS_FinOpsRunway"("quota_id", "is_latest");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "WITOS_FinOpsRunway_scope_type_scope_id_is_latest_idx" ON "WITOS_FinOpsRunway"("scope_type", "scope_id", "is_latest");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "WITOS_FinOpsScenario_scope_type_scope_id_idx" ON "WITOS_FinOpsScenario"("scope_type", "scope_id");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "WITOS_FinOpsAlertRule_enabled_idx" ON "WITOS_FinOpsAlertRule"("enabled");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "WITOS_FinOpsAlertEvent_status_created_at_idx" ON "WITOS_FinOpsAlertEvent"("status", "created_at");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "WITOS_FinOpsAlertEvent_alert_id_idx" ON "WITOS_FinOpsAlertEvent"("alert_id");
