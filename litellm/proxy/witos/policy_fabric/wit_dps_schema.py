"""WIT-DPS v2 JSON Schema (blueprint v2 §2.4).

The schema is the outer wall of the parsing security boundary: vendor documents
and customer imports are untrusted input, so `additionalProperties: false`
everywhere is deliberate. Anything the schema does not know about is surfaced
as `unmapped`, never silently accepted.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

_SCHEMA_JSON: Final = r"""
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://witone.one/schemas/wit-dps/v2.json",
  "title": "WIT-DPS v2 policy document",
  "type": "object",
  "required": ["wit_dps_version", "name", "mode", "condition", "actions"],
  "additionalProperties": false,
  "properties": {
    "wit_dps_version": {"const": "2.0"},
    "policy_id": {"type": "string", "maxLength": 128},
    "name": {"type": "string", "minLength": 1, "maxLength": 256},
    "description": {"type": "string", "maxLength": 4096},
    "severity": {"enum": ["low", "medium", "high", "critical"]},
    "mode": {"enum": ["mirror", "delegate", "hybrid", "observe"]},
    "priority": {"type": "integer", "minimum": 0, "maximum": 100000},
    "direction": {
      "enum": ["input", "output", "both", "tool", "tool_input", "tool_output", "rag_context"]
    },
    "source": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "vendor": {"type": "string", "maxLength": 64},
        "connection_id": {"type": "string", "maxLength": 128},
        "external_policy_id": {"type": "string", "maxLength": 256},
        "external_policy_version": {"type": "string", "maxLength": 64},
        "external_url": {"type": "string", "maxLength": 2048},
        "imported_at": {"type": "string", "maxLength": 64},
        "external_policy_hash": {"type": "string", "maxLength": 128},
        "canonical_policy_hash": {"type": "string", "maxLength": 128},
        "unmapped": {"type": "array", "items": {"type": "string", "maxLength": 512}, "maxItems": 512}
      }
    },
    "scope": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "entities": {
          "type": "array",
          "maxItems": 512,
          "items": {
            "type": "object",
            "required": ["type", "id"],
            "additionalProperties": false,
            "properties": {
              "type": {"enum": ["organization", "team", "user", "key_alias", "end_user"]},
              "id": {"type": "string", "minLength": 1, "maxLength": 256}
            }
          }
        },
        "models": {"type": "array", "items": {"type": "string", "maxLength": 256}, "maxItems": 512},
        "applications": {"type": "array", "items": {"type": "string", "maxLength": 256}, "maxItems": 512}
      }
    },
    "condition": {"$ref": "#/$defs/node"},
    "actions": {
      "type": "object",
      "required": ["on_match"],
      "additionalProperties": false,
      "properties": {
        "on_match": {
          "enum": ["BLOCK", "REQUIRE_APPROVAL", "REDACT", "MASK", "WARN", "AUDIT", "ALLOW"]
        },
        "redact_strategy": {"enum": ["mask", "replace", "hash", "remove"]},
        "block_message": {"type": "string", "maxLength": 1024},
        "alert_channels": {"type": "array", "items": {"type": "string", "maxLength": 256}, "maxItems": 64},
        "log_payload": {"enum": ["metadata_only", "none"]}
      }
    },
    "exceptions": {
      "type": "array",
      "maxItems": 512,
      "items": {
        "type": "object",
        "required": ["type", "value"],
        "additionalProperties": false,
        "properties": {
          "type": {
            "enum": ["identity", "group", "team", "organization", "application", "key_alias", "model"]
          },
          "value": {"type": "string", "minLength": 1, "maxLength": 256}
        }
      }
    },
    "streaming": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "mode": {"enum": ["buffer_full", "chunk_gate", "observe_only"]}
      }
    },
    "fail_mode": {"enum": ["fail_open", "fail_closed", "observe"]}
  },
  "$defs": {
    "node": {
      "oneOf": [
        {"$ref": "#/$defs/all_node"},
        {"$ref": "#/$defs/any_node"},
        {"$ref": "#/$defs/not_node"},
        {"$ref": "#/$defs/leaf"}
      ]
    },
    "all_node": {
      "type": "object",
      "required": ["operator", "conditions"],
      "additionalProperties": false,
      "properties": {
        "operator": {"const": "ALL"},
        "conditions": {"type": "array", "minItems": 1, "maxItems": 128, "items": {"$ref": "#/$defs/node"}}
      }
    },
    "any_node": {
      "type": "object",
      "required": ["operator", "conditions"],
      "additionalProperties": false,
      "properties": {
        "operator": {"const": "ANY"},
        "conditions": {"type": "array", "minItems": 1, "maxItems": 128, "items": {"$ref": "#/$defs/node"}}
      }
    },
    "not_node": {
      "type": "object",
      "required": ["operator", "condition"],
      "additionalProperties": false,
      "properties": {
        "operator": {"const": "NOT"},
        "condition": {"$ref": "#/$defs/node"}
      }
    },
    "leaf": {
      "oneOf": [
        {"$ref": "#/$defs/classifier_leaf"},
        {"$ref": "#/$defs/pattern_leaf"},
        {"$ref": "#/$defs/terms_leaf"},
        {"$ref": "#/$defs/context_leaf"},
        {"$ref": "#/$defs/tool_argument_leaf"}
      ]
    },
    "classifier_leaf": {
      "type": "object",
      "required": ["type"],
      "additionalProperties": false,
      "properties": {
        "type": {"enum": ["data_class", "sensitivity", "sensitivity_label", "classification_source"]},
        "class": {"type": "string", "minLength": 1, "maxLength": 256},
        "value": {"type": "string", "minLength": 1, "maxLength": 256},
        "min_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "min_count": {"type": "integer", "minimum": 1, "maximum": 10000}
      }
    },
    "pattern_leaf": {
      "type": "object",
      "required": ["type", "pattern"],
      "additionalProperties": false,
      "properties": {
        "type": {"const": "regex"},
        "pattern": {"type": "string", "minLength": 1, "maxLength": 4096},
        "case_sensitive": {"type": "boolean"},
        "min_count": {"type": "integer", "minimum": 1, "maximum": 10000},
        "classifier": {"type": "string", "maxLength": 256}
      }
    },
    "terms_leaf": {
      "type": "object",
      "required": ["type", "terms"],
      "additionalProperties": false,
      "properties": {
        "type": {"enum": ["dictionary", "keyword"]},
        "terms": {
          "type": "array",
          "minItems": 1,
          "maxItems": 5000,
          "items": {"type": "string", "minLength": 1, "maxLength": 512}
        },
        "case_sensitive": {"type": "boolean"},
        "whole_word": {"type": "boolean"},
        "min_count": {"type": "integer", "minimum": 1, "maximum": 10000},
        "classifier": {"type": "string", "maxLength": 256}
      }
    },
    "context_leaf": {
      "type": "object",
      "required": ["type", "value"],
      "additionalProperties": false,
      "properties": {
        "type": {
          "enum": [
            "identity", "group", "team", "organization", "application",
            "model", "model_group", "provider", "destination", "file_type", "tool"
          ]
        },
        "value": {"type": "string", "minLength": 1, "maxLength": 512}
      }
    },
    "tool_argument_leaf": {
      "type": "object",
      "required": ["type", "argument"],
      "additionalProperties": false,
      "properties": {
        "type": {"const": "tool_argument"},
        "tool": {"type": "string", "maxLength": 256},
        "argument": {"type": "string", "minLength": 1, "maxLength": 256},
        "value": {"type": "string", "maxLength": 512}
      }
    }
  }
}
"""


def _load_schema(raw: str) -> Mapping[str, object]:
    parsed: Final[object] = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("WIT-DPS schema must decode to a JSON object")
    return MappingProxyType(parsed)


WIT_DPS_V2_SCHEMA: Final[Mapping[str, object]] = _load_schema(_SCHEMA_JSON)
