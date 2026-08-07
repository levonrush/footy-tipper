# Independent delivery watchdog

GitHub scheduled events are best-effort and may be delayed or dropped. Footy
Tipper therefore uses two clocks:

- GitHub polls at off-boundary Sydney times from 11:07 through 14:37.
- A Cloudflare Worker independently requests the same gate at 11:27, 11:57,
  and every 30 minutes through 14:57.

Neither clock can request production `live` directly. Both call the Drive-backed
gate, and every eventual live run shares the same Actions concurrency lock,
Drive delivery marker, and SQLite send ledger.

## One-time setup

### 1. Create the repository-scoped GitHub App

In GitHub **Settings → Developer settings → GitHub Apps**, create an app with:

- a unique name such as `footy-tipper-delivery-watchdog-levonrush`;
- webhooks disabled;
- repository permission **Actions: Read and write**;
- no account permissions and no subscribed events;
- installation restricted to the owner account.

Generate one private key. Install the app using **Only select repositories** and
select only `footy-tipper`.

Record the numeric **App ID**, the installation ID from the installation URL,
and the app slug. The workflow actor is `<app-slug>[bot]`. Configure that
non-secret identity in GitHub:

```bash
gh variable set FOOTY_TIPPER_WATCHDOG_ACTOR \
  --repo levonrush/footy-tipper \
  --body '<app-slug>[bot]'
```

The workflow fails closed if this variable is absent or a different actor sets
the internal watchdog input. Bot actors are also refused from the human
`test`, `refresh`, and `live` dispatch branch.

### 2. Configure and deploy the Worker

Create or sign in to a Cloudflare account with Workers Free enabled. In
[`watchdog/wrangler.jsonc`](../watchdog/wrangler.jsonc), replace only:

- `REPLACE_WITH_GITHUB_APP_ID`
- `REPLACE_WITH_GITHUB_INSTALLATION_ID`

Then run:

```bash
cd watchdog
npm ci --ignore-scripts
npx wrangler login
npx wrangler secret put GITHUB_APP_PRIVATE_KEY
npm run deploy
```

Paste the complete downloaded PEM only at Wrangler's secret prompt. Cloudflare
stores it encrypted; the value must never be committed or placed in a normal
Worker variable.

### 3. Verify without sending

Before the first watchdog dispatch, confirm that the real production gate is
not currently `live`:

```bash
footy-tipper advanced cloud gate
```

When it reports `skip`, wait for the next Cloudflare recovery tick within the
11:00 to 14:59 Sydney window. The resulting Actions run must be named
`Predict watchdog gate`, must show the configured app actor, and must finish
after the gate with no prediction job, email, Drive marker, or ledger change.

Also verify:

```bash
footy-tipper status --json
```

The next round must remain unsent, with no unresolved delivery marker. On the
next actual first-game day, GitHub should normally start the run at 11:07; if
it does not, the watchdog should create a gated run at 11:27.

## Failure and recovery

An automated gate or prediction failure creates or updates one assigned GitHub
issue labelled `automation-alert`. The issue contains only the run URL, actor,
gate result, mode, reason, and prediction result. A later successful live run
closes it.

Always begin with:

```bash
footy-tipper status
```

A `pending` Drive marker means SMTP may have delivered partially. It blocks both
clocks and must be reconciled from SMTP evidence; never delete it or force an
automatic resend.

Cloudflare retries transient GitHub network, rate-limit, conflict, and server
errors. Permanent authorization errors fail the invocation without logging the
private key, JWT, installation token, or GitHub response body.

## Key rotation

GitHub App private keys do not expire automatically. To rotate without downtime:

1. Generate a second GitHub App private key.
2. Replace the Worker secret with `npx wrangler secret put GITHUB_APP_PRIVATE_KEY`.
3. Verify a guarded `skip` dispatch.
4. Delete the old key from the GitHub App.

## Rollback

Disable or delete the Worker's Cron Trigger in Cloudflare first. GitHub's
Sydney-time schedule remains active and delivery state is unchanged. If the
watchdog is being retired, uninstall the GitHub App, delete its private keys,
and remove `FOOTY_TIPPER_WATCHDOG_ACTOR` only after the cron is disabled.
