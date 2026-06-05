"""Source downloaders. Each is resumable, rate-limited, cached, provenance-tagged.

Phase 0: stubs only. Phase 1+ implements per source. Existing
`crimellm.corpora` downloaders (USC, CourtListener, UK leg) feed into these
adapters until per-source modules land.
"""
