"""Saved scenarios (blueprint §1.10, §1.11).

Only the assumptions are stored, never the evaluated numbers. A scenario is a
question, and the answer depends on the forecast and the price book it is asked
against: caching last week's answer would hand a CFO a saving that quietly
stopped being true when the underlying forecast moved.
"""

import json
from datetime import datetime
from typing import Final, TypedDict

from pydantic import TypeAdapter
from typing_extensions import ReadOnly

from litellm.proxy.witos.finops.history import ScopeRef
from litellm.proxy.witos.finops.scenarios import ScenarioAssumptions
from litellm.proxy.witos.shared.sql import SqlExecutor


class ScenarioRow(TypedDict):
    scenario_id: ReadOnly[str]
    name: ReadOnly[str]
    description: ReadOnly[str | None]
    scope_type: ReadOnly[str]
    scope_id: ReadOnly[str]
    assumptions_json: ReadOnly[object]
    created_by: ReadOnly[str]
    created_at: ReadOnly[datetime]


_SCENARIO_ROWS: Final = TypeAdapter(tuple[ScenarioRow, ...])

_COLUMNS: Final = "scenario_id, name, description, scope_type, scope_id, assumptions_json, created_by, created_at"

_LIST_SQL: Final = f"""
SELECT {_COLUMNS} FROM "WITOS_FinOpsScenario"
WHERE ($1::text IS NULL OR scope_type = $1) AND ($2::text IS NULL OR scope_id = $2)
ORDER BY created_at DESC
"""

_BY_ID_SQL: Final = f'SELECT {_COLUMNS} FROM "WITOS_FinOpsScenario" WHERE scenario_id = $1'

_INSERT_SQL: Final = f"""
INSERT INTO "WITOS_FinOpsScenario" (
    scenario_id, name, description, scope_type, scope_id, assumptions_json, created_by, created_at, updated_at
)
VALUES (
    gen_random_uuid()::text, $1, $2, $3, $4, $5::jsonb, $6,
    (now() AT TIME ZONE 'UTC'), (now() AT TIME ZONE 'UTC')
)
RETURNING {_COLUMNS}
"""

_DELETE_SQL: Final = 'DELETE FROM "WITOS_FinOpsScenario" WHERE scenario_id = $1'


class ScenarioStore:
    def __init__(self, executor: SqlExecutor) -> None:
        self._executor: Final = executor

    async def list(self, scope: ScopeRef | None = None) -> tuple[ScenarioRow, ...]:
        return _SCENARIO_ROWS.validate_python(
            await self._executor.query(
                _LIST_SQL, scope.scope_type if scope else None, scope.scope_id if scope else None
            )
        )

    async def by_id(self, scenario_id: str) -> ScenarioRow | None:
        rows: Final = _SCENARIO_ROWS.validate_python(await self._executor.query(_BY_ID_SQL, scenario_id))
        return rows[0] if rows else None

    async def create(
        self,
        *,
        name: str,
        description: str | None,
        scope: ScopeRef,
        assumptions: ScenarioAssumptions,
        created_by: str,
    ) -> ScenarioRow:
        rows: Final = _SCENARIO_ROWS.validate_python(
            await self._executor.query(
                _INSERT_SQL,
                name,
                description,
                scope.scope_type,
                scope.scope_id,
                assumptions.model_dump_json(),
                created_by,
            )
        )
        return rows[0]

    async def delete(self, scenario_id: str) -> bool:
        return await self._executor.execute(_DELETE_SQL, scenario_id) > 0


def assumptions_of(row: ScenarioRow) -> ScenarioAssumptions:
    """Parse a stored assumption document, whichever way the driver returned it."""
    raw: Final = row["assumptions_json"]
    if isinstance(raw, str):
        return ScenarioAssumptions.model_validate(json.loads(raw))
    return ScenarioAssumptions.model_validate(raw)
