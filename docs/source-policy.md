# Source and content-use policy

Footy Tipper stores structured facts needed to reproduce a prediction. It does not treat access to a web page as permission to republish it. Provider terms, licence scope, temporal provenance, and the smallest useful stored representation are part of the data contract.

This is an engineering policy, not legal advice. Terms can change; the maintainer must review the current provider terms before changing collection scope or monetising the product.

## Principles

1. Use documented feeds, APIs, licensed files, and official confirmations where possible.
2. Store normalized facts, identifiers, timestamps, hashes, and source links rather than third-party prose or media.
3. Preserve when a fact was observed so later information cannot leak into an earlier prediction.
4. Do not defeat access controls, authentication boundaries, robots controls, rate limits, or technical restrictions.
5. A source outage fails soft only where the downstream contract explicitly supports a missing value; it must never be disguised as fresh data.
6. Re-check terms and attribution requirements before any commercial or wider public deployment.

## Active source inventory

| Source family | Purpose | Stored form | Operational boundary |
| --- | --- | --- | --- |
| nrl.com draw and match centres | Fixtures, results, ladder/performance inputs | Structured cache rows and source provenance | Python ingestion; legacy XML is explicit rollback |
| nrl.com Team Lists and Late Mail | Versioned lineups | Article snapshot metadata plus normalized lineup rows | Historical as-of and live latest-known selection |
| Australia Sports Betting | Historical markets | Normalized odds observations | Fill historical gaps; retain observation type/time |
| The Odds API / Betfair | Live pre-game markets | Versioned odds observations | Credentials, jurisdiction, freshness, and no-market fallback apply |
| ABC RSS / GDELT | Club Context candidate discovery | URL, title/identifier, publisher, publication or first-observed time, and hash | Discovery only; never sufficient confirmation |
| Official NRL/club or reputable reporting | Club Context confirmation | Factual summary, link, timestamps, evidence role, and rights state | Approved facts-and-links events only; no article body |

The feed-specific runtime, parity, and rollback contracts remain in [Data-source migration](data-source-migration.md). Club Context's evidence rules are in [Club Context](club-context.md).

Club Context discovery is deliberately browser-free: the optional ABC RSS and GDELT JSON adapters use Python's standard HTTP/XML/JSON libraries. Chrome, Chrome Headless Shell, Playwright, Selenium, and browser-driver installation are not runtime dependencies.

## Club Context rights states

Every context source adapter and snapshot carries one explicit state:

| State | Meaning |
| --- | --- |
| `facts_and_links` | Approved to retain a concise independently written factual summary, provenance fields, and a link |
| `licensed` | Use is covered by a documented licence whose scope includes the operation |
| `discovery_only` | May identify a candidate/link; its content is not extracted or treated as event evidence |
| `prohibited` | Adapter must not run and the source cannot support an eligible event |
| `unknown` | Disabled for automated use until reviewed |

The acquisition method (`adapter`, `manual`, or `import`) is stored separately from rights. A manually entered article is not automatically safe to reuse, and an open discovery index does not grant rights over the underlying publishers' articles.

## Official-first confirmation

For Club Context, ABC RSS and GDELT are observation channels. GDELT describes its own datasets as available for unrestricted use with attribution, but that does not transfer rights in the publisher content it indexes. The [GDELT terms](https://www.gdeltproject.org/about.html#termsofuse), the [ABC terms](https://www.abc.net.au/conditions.htm), and the [NRL terms](https://www.nrl.com/terms-of-use) must be evaluated independently.

An event can reach the eligible registry only through:

- one official NRL or club confirmation; or
- two reputable independent confirmations.

Rumours, anonymous claims, social-media inference, duplicated syndication, and low-confidence classification stay pending or are rejected. Sensitive-event evidence receives the same truth gate and stricter copy handling; sensitivity never becomes permission to infer medical details.

## What may be persisted

Club Context article snapshots retain only:

- canonical URL and publisher/source key;
- publication time and first-observed time;
- title or stable source identifier where required for review;
- content hash and extraction/version metadata;
- rights status, acquisition method, and terms link; and
- source evidence relationships to a separately written factual event summary.

They do not retain article bodies, long extracts, images, audio, video, or generated embeddings of publisher text. Reader cards link back to the supporting source. Changing a summary creates new reviewed state rather than silently rewriting the immutable prediction snapshot that used it.

## Adapter and outage rules

- An adapter whose rights state is `prohibited` or `unknown` is disabled.
- Discovery and extraction permissions are separate. A discovery-allowed adapter may still be extraction-disabled.
- Rate limits, timeouts, parsing failures, and missing optional dependencies are recorded in `context_ingestion_runs`.
- Default weekly ingestion fails soft and retains the last valid registry state. Explicit strict diagnosis may return non-zero.
- No adapter failure can alter an existing probability or suppress delivery.
- Coverage reports distinguish “no event” from “source unavailable”; absence of coverage is not evidence that nothing happened.

## Review and monetisation gate

Before enabling a new source or materially changing an adapter, record:

1. source owner and canonical terms/licence URL;
2. permitted discovery, extraction, storage, attribution, redistribution, and commercial use;
3. authentication, rate-limit, retention, and deletion requirements;
4. fields persisted and why each is necessary;
5. an outage/withdrawal path; and
6. review date and reviewer.

Before monetised deployment, repeat this review for nrl.com, clubs, ABC, GDELT, odds providers, email/site use, and every active adapter. `facts_and_links` is a conservative engineering control, not a substitute for that review.
