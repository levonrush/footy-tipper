import unittest

from pipeline.common.club_context.discovery import discover_all
from pipeline.common.club_context.sources import DiscoveryCandidate, SourcePolicy
from pipeline.common.club_context.taxonomy import RightsStatus, SourceClass


class _WorkingAdapter:
    policy = SourcePolicy(
        source_key="working",
        source_class=SourceClass.PUBLIC_BROADCASTER,
        rights_status=RightsStatus.DISCOVERY_ONLY,
        automated_discovery_allowed=True,
        automated_extraction_allowed=False,
    )

    def discover(self, **_kwargs):
        return [
            DiscoveryCandidate(
                url="https://example.com/verified-candidate",
                title="Candidate headline",
                publisher_key="example.com",
                source_class=SourceClass.PUBLIC_BROADCASTER,
                snippet="Discovery text that must not become event truth.",
            )
        ]


class _UnavailableAdapter:
    policy = SourcePolicy(
        source_key="unavailable",
        source_class=SourceClass.DISCOVERY_INDEX,
        rights_status=RightsStatus.DISCOVERY_ONLY,
        automated_discovery_allowed=True,
        automated_extraction_allowed=False,
    )

    def discover(self, **_kwargs):
        raise TimeoutError("source timed out")


class _RightsDisabledAdapter:
    policy = SourcePolicy(
        source_key="rights-disabled",
        source_class=SourceClass.REPUTABLE_MEDIA,
        rights_status=RightsStatus.PROHIBITED,
        automated_discovery_allowed=False,
        automated_extraction_allowed=False,
    )

    def __init__(self):
        self.called = False

    def discover(self, **_kwargs):
        self.called = True
        raise AssertionError("rights-disabled adapter must not run")


class ClubContextDiscoveryTests(unittest.TestCase):
    def test_one_source_outage_does_not_discard_other_candidates(self):
        candidates, errors = discover_all(
            query="NRL context",
            adapters=[_UnavailableAdapter(), _WorkingAdapter()],
        )

        self.assertEqual(len(candidates), 1)
        policy, candidate = candidates[0]
        self.assertEqual(policy.source_key, "working")
        self.assertEqual(candidate.url, "https://example.com/verified-candidate")
        self.assertEqual(errors, ["unavailable: source timed out"])

    def test_rights_disabled_adapter_is_not_called(self):
        disabled = _RightsDisabledAdapter()
        candidates, errors = discover_all(
            query="NRL context",
            adapters=[disabled, _WorkingAdapter()],
        )

        self.assertFalse(disabled.called)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(errors, [])

    def test_duplicate_urls_are_collapsed_across_adapters(self):
        candidates, errors = discover_all(
            query="NRL context",
            adapters=[_WorkingAdapter(), _WorkingAdapter()],
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
