#!/usr/bin/env python3
"""
Template-based update_services.py for Inception.

Yields model dictionaries that are rendered using Jinja2 templates.

Usage: python scripts/update_services.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Iterator

import httpx

from unitysvc_sellers.model_data import ModelDataFetcher, ModelDataLookup
from unitysvc_sellers.params_render import write_params_from_iterator

# Provider Configuration
PROVIDER_NAME = "inception"
PROVIDER_DISPLAY_NAME = "Inception"
API_BASE_URL = "https://api.inceptionlabs.ai/v1"
ENV_API_KEY_NAME = "INCEPTION_API_KEY"

SCRIPT_DIR = Path(__file__).parent
SPECS_DIR = SCRIPT_DIR.parent / "specs"


def committed_parameters(service_name: str) -> dict:
    """The parameters already committed for ``service_name`` ({} if it is new).

    unitysvc-sellers >= 0.3.1 keeps a committed value when the iterator yields
    ``None`` for it: from inside the writer, a lookup that failed and a lookup
    that legitimately found nothing are the same event. That is right for
    enrichment, but it means a price we FAILED to derive gets re-shipped as
    though it were this run's answer. Reading the previous value here is what
    separates the two cases — see the price guard in ``_build_template_vars``.
    """
    path = SPECS_DIR / f"{service_name}.json"
    if not path.is_file():
        return {}
    try:
        return (json.loads(path.read_text()) or {}).get("parameters") or {}
    except (OSError, ValueError):
        return {}


class ModelSource:
    """Fetches models and yields template dictionaries."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.data_fetcher = ModelDataFetcher()
        self.litellm_data = None

    def iter_models(self) -> Iterator[dict]:
        """Yield model dictionaries for template rendering."""
        # Fetch LiteLLM data once
        self.litellm_data = self.data_fetcher.fetch_litellm_model_data()
        if not self.litellm_data:
            print(
                "Error: LiteLLM model data came back empty. Every price lookup "
                "would fail, and unitysvc-sellers >= 0.3.1 would preserve the "
                "committed prices instead — re-shipping stale rate cards as "
                "though they were current."
            )
            sys.exit(1)

        print(f"Fetching models from {PROVIDER_DISPLAY_NAME} API...")
        try:
            r = httpx.get(
                f"{API_BASE_URL}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0,
            )
            r.raise_for_status()
            models = r.json().get("data", [])
            print(f"Found {len(models)} models\n")
        except Exception as e:
            print(f"Error listing models: {e}")
            # Not `return`. An empty iterator is indistinguishable from "the
            # upstream retired its whole catalog": with deprecate_missing the
            # writer would mark every committed service deprecated, and exiting
            # 0 would make a failed fetch look like a clean no-change run.
            sys.exit(1)

        if not models:
            print(
                "Error: upstream enumerated zero models — refusing to treat an "
                "empty enumeration as a retired catalog."
            )
            sys.exit(1)

        for i, model_info in enumerate(models, 1):
            model_id = model_info.get("id", "")
            print(f"[{i}/{len(models)}] {model_id}")

            # Build template variables
            template_vars = self._build_template_vars(model_id, model_info)
            if template_vars:
                yield template_vars
                print("  OK")

    # Inception's "edit" models target a custom /v1/edit/completions
    # endpoint with a structured prompt format (see
    # https://docs.inceptionlabs.ai/capabilities/next-edit.md), not the
    # standard /chat/completions surface.  The platform gateway, the
    # ``llm`` service_type, and the ``llm_*`` presets all assume
    # /chat/completions, so advertising these as standard LLM offerings
    # makes connectivity probes and code examples fail with upstream 500.
    # Re-enable when /edit/completions routing + an "edit" service_type
    # / preset land — tracked at https://github.com/unitysvc/unitysvc.
    EDIT_MODEL_IDS = frozenset({"mercury-edit", "mercury-edit-2"})

    def _build_template_vars(self, model_id: str, model_info: dict) -> dict | None:
        """Build template variables for a model."""
        if model_id in self.EDIT_MODEL_IDS:
            print(f"  Skipped: {model_id} uses /edit/completions; not supported yet")
            return None

        service_name = f"{PROVIDER_NAME}/{model_id}"
        service_type = self._determine_service_type(model_id)
        display_name = model_id.replace("-", " ").replace("_", " ").title()

        # Build details from LiteLLM data and model info
        details = {}
        model_data = ModelDataLookup.lookup_model_details(
            model_id, self.litellm_data or {})

        if model_data:
            for field in [
                    "max_tokens", "max_input_tokens", "max_output_tokens",
                    "mode"
            ]:
                if field in model_data:
                    details[field] = model_data[field]
            if "litellm_provider" in model_data:
                details["litellm_provider"] = model_data["litellm_provider"]

        if "owned_by" in model_info:
            details["owned_by"] = model_info["owned_by"]
        if "object" in model_info:
            details["object"] = model_info["object"]

        # Inception's /v1/models enumerates what each model can actually do:
        #   "supported_features": ["tools", "json_mode", "structured_outputs"]
        # `tools` is the one the catalog could not previously say, and saying it
        # wrongly is expensive -- a service that advertises tool calling without
        # it answers the shipped example with `400 invalid request: tool use is
        # not supported by the provided model`.  Carry the upstream list through
        # verbatim so offering.json.j2 can derive `feature:func-call` from the
        # provider's own answer rather than from an assumption about the
        # catalog.  Absent on an older upstream => no tag, which under-claims.
        if model_info.get("supported_features"):
            details["supported_features"] = sorted(
                model_info["supported_features"])

        # Canonical (snake_case) metadata required by the platform validator
        # for LLM offerings.  Both keys must be present; null asserts
        # "unknown".  Claude models are closed-source so parameter_count
        # is permanently null per the canonical helper.  metadata_sources
        # records provenance so reviewers can triage stale-value reports.
        canonical = ModelDataLookup.get_canonical_metadata(
            model_id,
            fetcher=self.data_fetcher,
        )
        details["context_length"] = canonical["context_length"]
        details["parameter_count"] = canonical["parameter_count"]
        if canonical["sources"]:
            details["metadata_sources"] = canonical["sources"]

        # BYOK: the customer supplies their own API key, so usage is billed by
        # the provider directly and UnitySVC meters nothing — the price is Free.
        # This plain description is what payout_price keeps (seller-facing). The
        # customer-facing listing cell is composed in listing.json.j2 from
        # pricing_note, into the "<amount> ~ <PILL> | <note>" grammar; do not
        # build it here, since this dict feeds payout_price too.
        pricing = {
            "type": "constant",
            "price": "0",
            "description": "Free (BYOK)",
        }
        pricing_note = None
        if model_data and "input_cost_per_token" in model_data and "output_cost_per_token" in model_data:
            input_price = round(float(
                model_data["input_cost_per_token"]) * 1_000_000, 4)
            output_price = round(float(
                model_data["output_cost_per_token"]) * 1_000_000, 4)
            if "cache_read_input_token_cost" in model_data:
                cached_price = round(float(
                    model_data["cache_read_input_token_cost"]) * 1_000_000, 4)
                pricing_note = (
                    f"${self._format_price(input_price)} / "
                    f"${self._format_price(output_price)} / "
                    f"${self._format_price(cached_price)} "
                    f"per 1M input/output/cached tokens"
                )
            else:
                pricing_note = (
                    f"${self._format_price(input_price)} / "
                    f"${self._format_price(output_price)} "
                    f"per 1M input/output tokens"
                )

        # `pricing_note` is the only field derived from the upstream rate card
        # here (the price itself is the constant "Free (BYOK)"). It is nullable,
        # it is a template param rather than a schema field, and so a failed
        # lookup is rejected by nothing downstream. Since unitysvc-sellers 0.3.1
        # preserves committed values against a yielded None, that failure now
        # SHIPS THE PREVIOUS RATE CARD as though it were this run's answer. A
        # model that has never appeared in the LiteLLM data has no committed
        # value and nothing to silently ship; a model that had one and can no
        # longer derive it is the regression, and it is fatal.
        if pricing_note is None and committed_parameters(service_name).get("pricing_note") is not None:
            print(
                f"  FATAL: {model_id} has a committed pricing_note but no "
                "input_cost_per_token/output_cost_per_token in this run's "
                "LiteLLM data. Refusing to re-ship the previous rate card."
            )
            sys.exit(1)

        return {
            # The service's name IS its path under specs/ (flat layout, #1263).
            # unitysvc-sellers >= 0.3.1 requires this key verbatim: `name_field`
            # is gone and there is no fallback for a dict that omits it.
            "service_name": service_name,
            # Offering name is the bare upstream model_id
            "offering_name": model_id,
            # Offering fields
            "display_name": display_name,
            "description": f"{display_name} language model",
            "service_type": service_type,
            "status": "ready",
            "details": details,
            "payout_price": pricing,
            # Reference rates for the BYOK pricing paragraph (template-rendered)
            "pricing_note": pricing_note,
            # Listing fields
            "list_price": pricing,
            # Provider config (for templates)
            "provider_name": PROVIDER_NAME,
            "provider_display_name": PROVIDER_DISPLAY_NAME,
            "api_base_url": API_BASE_URL,
            "env_api_key_name": ENV_API_KEY_NAME,
        }

    def _determine_service_type(self, model_id: str) -> str:
        model_lower = model_id.lower()
        if any(kw in model_lower for kw in ["embed", "embedding"]):
            return "embedding"
        if any(kw in model_lower for kw in ["rerank"]):
            return "rerank"
        if any(kw in model_lower for kw in ["vision"]):
            return "vision_language_model"
        return "llm"

    def _format_price(self, price: float) -> str:
        """Format price without trailing .0 for whole numbers."""
        if price == int(price):
            return str(int(price))
        return str(price)


def main():
    api_key = os.environ.get(ENV_API_KEY_NAME)
    if not api_key:
        print(f"Error: {ENV_API_KEY_NAME} not set")
        sys.exit(1)

    source = ModelSource(api_key)
    write_params_from_iterator(
        iterator=source.iter_models(),
        output_dir=SPECS_DIR,
    )


if __name__ == "__main__":
    main()
