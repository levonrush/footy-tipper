# Delivery watchdog

This directory contains the Google Apps Script backup clock for scheduled
production tips. The script never predicts or sends email itself. During the
Sydney recovery window it dispatches `predict.yml` with only the guarded
`watchdog` input; GitHub then reads the normal Drive schedule.

The production project was installed and its first automatic heartbeat was
verified on 7 August 2026. This directory remains the deployable source and
test suite; it is not a second runtime.

## Local verification

Requires Node.js 20 or newer.

```bash
npm ci --ignore-scripts
npm run check
npm test
npm audit --audit-level=high
```

The tests execute the deployable Apps Script source with mocked Google
services. They cover AEST/AEDT, recovery boundaries, successful-slot
idempotency, retries, error redaction, exact dispatch payload, actor
validation, trigger replacement, and missing credentials.

## Deployment

`@google/clasp` is pinned as development tooling. For a complete rebuild after
Google authorization:

```bash
npm run login
npm run create
npm run push
npm run open
```

The generated `.clasp.json` is ignored. Store the repository-scoped GitHub
token only as the Apps Script Property `GITHUB_TOKEN`; never place it in this
directory or a command line.

See [watchdog setup and recovery](../docs/watchdog-setup.md) for the complete
deployed-state record, one-time authorization, verification, credential
replacement, incident handling, and rollback procedure.
