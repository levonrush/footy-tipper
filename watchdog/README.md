# Delivery watchdog

This Cloudflare Worker is the independent clock for scheduled production tips.
It never predicts or sends email itself. During the Sydney recovery window it
authenticates as a repository-scoped GitHub App and dispatches `predict.yml`
with the guarded `watchdog` input. The GitHub workflow then reads the normal
Drive schedule and chooses only `live`, `refresh`, or `skip`.

The Worker runs at minutes 27 and 57 of every UTC hour, but dispatches only
between 11:00 and 14:59 `Australia/Sydney`. The timezone check handles AEST and
AEDT. GitHub remains the primary clock at separate minute offsets.

## Local verification

Requires Node.js 22 or newer.

```bash
npm ci --ignore-scripts
npm run check
npm test
npm audit --audit-level=high
```

`wrangler.jsonc` deliberately contains placeholders for the GitHub App and
installation IDs. Replace those two non-secret values during the one-time
production setup. Never add the private key to that file, `.dev.vars`, a shell
history entry, or Git.

## Production configuration

The deployed Worker requires:

| Binding | Type | Purpose |
| --- | --- | --- |
| `GITHUB_OWNER` | plain variable | `levonrush` |
| `GITHUB_REPO` | plain variable | `footy-tipper` |
| `GITHUB_WORKFLOW_FILE` | plain variable | `predict.yml` |
| `GITHUB_REF` | plain variable | `main` |
| `GITHUB_APP_ID` | plain variable | numeric GitHub App ID |
| `GITHUB_INSTALLATION_ID` | plain variable | selected-repository installation ID |
| `GITHUB_APP_PRIVATE_KEY` | encrypted secret | non-expiring GitHub App private key |

The GitHub App must be installed only on `footy-tipper` with repository
`Actions: Read and write`; metadata read access is implicit. It needs no
webhook, user authorization, contents permission, SMTP credential, Google
credential, or recipient data.

See [watchdog setup and recovery](../docs/watchdog-setup.md) for the complete
one-time deployment and rollback procedure.
