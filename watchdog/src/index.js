const API_VERSION = "2026-03-10";
const USER_AGENT = "footy-tipper-cloudflare-watchdog";
const SYDNEY_TIMEZONE = "Australia/Sydney";
const DISPATCH_MINUTES = new Set([27, 57]);
const FIRST_DISPATCH_HOUR = 11;
const LAST_DISPATCH_HOUR = 14;
const RETRY_DELAYS_MS = [500, 2000];

class RetryableError extends Error {}

function encodeBase64Url(value) {
  const bytes =
    typeof value === "string" ? new TextEncoder().encode(value) : new Uint8Array(value);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/u, "");
}

function decodePem(pem) {
  const text = String(pem || "").trim();
  const match = text.match(
    /^-----BEGIN (RSA )?PRIVATE KEY-----([\s\S]+)-----END (RSA )?PRIVATE KEY-----$/u,
  );
  if (!match || Boolean(match[1]) !== Boolean(match[3])) {
    throw new Error("GITHUB_APP_PRIVATE_KEY is not a supported PEM private key.");
  }
  const binary = atob(match[2].replace(/\s+/gu, ""));
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return { bytes, pkcs1: Boolean(match[1]) };
}

function derLength(length) {
  if (length < 0x80) return Uint8Array.of(length);
  const bytes = [];
  let remaining = length;
  while (remaining > 0) {
    bytes.unshift(remaining & 0xff);
    remaining >>= 8;
  }
  return Uint8Array.of(0x80 | bytes.length, ...bytes);
}

function concatBytes(...parts) {
  const length = parts.reduce((total, part) => total + part.length, 0);
  const result = new Uint8Array(length);
  let offset = 0;
  for (const part of parts) {
    result.set(part, offset);
    offset += part.length;
  }
  return result;
}

function wrapPkcs1AsPkcs8(pkcs1) {
  const version = Uint8Array.of(0x02, 0x01, 0x00);
  const rsaAlgorithmIdentifier = Uint8Array.of(
    0x30,
    0x0d,
    0x06,
    0x09,
    0x2a,
    0x86,
    0x48,
    0x86,
    0xf7,
    0x0d,
    0x01,
    0x01,
    0x01,
    0x05,
    0x00,
  );
  const privateKey = concatBytes(Uint8Array.of(0x04), derLength(pkcs1.length), pkcs1);
  const body = concatBytes(version, rsaAlgorithmIdentifier, privateKey);
  return concatBytes(Uint8Array.of(0x30), derLength(body.length), body);
}

async function importPrivateKey(pem) {
  const decoded = decodePem(pem);
  const pkcs8 = decoded.pkcs1 ? wrapPkcs1AsPkcs8(decoded.bytes) : decoded.bytes;
  return crypto.subtle.importKey(
    "pkcs8",
    pkcs8,
    {
      name: "RSASSA-PKCS1-v1_5",
      hash: "SHA-256",
    },
    false,
    ["sign"],
  );
}

export async function createAppJwt(appId, privateKeyPem, nowMs = Date.now()) {
  if (!/^\d+$/u.test(String(appId || ""))) {
    throw new Error("GITHUB_APP_ID must be configured as a numeric GitHub App ID.");
  }
  const issuedAt = Math.floor(nowMs / 1000) - 60;
  const header = encodeBase64Url(JSON.stringify({ alg: "RS256", typ: "JWT" }));
  const payload = encodeBase64Url(
    JSON.stringify({
      iat: issuedAt,
      exp: issuedAt + 9 * 60,
      iss: String(appId),
    }),
  );
  const unsigned = `${header}.${payload}`;
  const key = await importPrivateKey(privateKeyPem);
  const signature = await crypto.subtle.sign(
    "RSASSA-PKCS1-v1_5",
    key,
    new TextEncoder().encode(unsigned),
  );
  return `${unsigned}.${encodeBase64Url(signature)}`;
}

export function sydneyClock(scheduledTimeMs) {
  const parts = new Intl.DateTimeFormat("en-AU", {
    timeZone: SYDNEY_TIMEZONE,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hourCycle: "h23",
  }).formatToParts(new Date(scheduledTimeMs));
  return Object.fromEntries(
    parts
      .filter((part) => part.type !== "literal")
      .map((part) => [part.type, Number(part.value)]),
  );
}

export function isDispatchWindow(scheduledTimeMs) {
  const clock = sydneyClock(scheduledTimeMs);
  return (
    clock.hour >= FIRST_DISPATCH_HOUR &&
    clock.hour <= LAST_DISPATCH_HOUR &&
    DISPATCH_MINUTES.has(clock.minute)
  );
}

function requiredEnvironment(env) {
  const required = [
    "GITHUB_OWNER",
    "GITHUB_REPO",
    "GITHUB_WORKFLOW_FILE",
    "GITHUB_REF",
    "GITHUB_APP_ID",
    "GITHUB_INSTALLATION_ID",
    "GITHUB_APP_PRIVATE_KEY",
  ];
  const values = Object.fromEntries(
    required.map((name) => [name, String(env?.[name] || "").trim()]),
  );
  const missing = required.filter((name) => !values[name]);
  if (missing.length) {
    throw new Error(`Watchdog configuration is missing: ${missing.join(", ")}.`);
  }
  if (!/^\d+$/u.test(values.GITHUB_INSTALLATION_ID)) {
    throw new Error("GITHUB_INSTALLATION_ID must be configured as a numeric ID.");
  }
  return values;
}

function apiHeaders(token) {
  return {
    Accept: "application/vnd.github+json",
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
    "User-Agent": USER_AGENT,
    "X-GitHub-Api-Version": API_VERSION,
  };
}

function isRetryableStatus(status) {
  return status === 408 || status === 409 || status === 429 || status >= 500;
}

async function githubRequest(fetchImpl, operation, url, options) {
  let response;
  try {
    response = await fetchImpl(url, options);
  } catch (error) {
    throw new RetryableError(`${operation} failed because the network request failed.`, {
      cause: error,
    });
  }
  if (!response.ok) {
    const message = `${operation} failed with GitHub status ${response.status}.`;
    if (isRetryableStatus(response.status)) throw new RetryableError(message);
    throw new Error(message);
  }
  if (response.status === 204) return {};
  const text = await response.text();
  return text ? JSON.parse(text) : {};
}

async function wait(delayMs) {
  await new Promise((resolve) => setTimeout(resolve, delayMs));
}

export async function withRetries(operation, delays = RETRY_DELAYS_MS) {
  for (let attempt = 0; ; attempt += 1) {
    try {
      return await operation(attempt + 1);
    } catch (error) {
      if (!(error instanceof RetryableError) || attempt >= delays.length) throw error;
      await wait(delays[attempt]);
    }
  }
}

export async function dispatchGate(env, fetchImpl = fetch, nowMs = Date.now()) {
  const config = requiredEnvironment(env);
  return withRetries(async () => {
    const appJwt = await createAppJwt(
      config.GITHUB_APP_ID,
      config.GITHUB_APP_PRIVATE_KEY,
      nowMs,
    );
    const installation = await githubRequest(
      fetchImpl,
      "GitHub App installation token request",
      `https://api.github.com/app/installations/${config.GITHUB_INSTALLATION_ID}/access_tokens`,
      {
        method: "POST",
        headers: apiHeaders(appJwt),
        body: JSON.stringify({
          repositories: [config.GITHUB_REPO],
          permissions: { actions: "write" },
        }),
      },
    );
    if (!installation.token) {
      throw new Error("GitHub App installation token response did not contain a token.");
    }

    return githubRequest(
      fetchImpl,
      "GitHub workflow dispatch",
      `https://api.github.com/repos/${config.GITHUB_OWNER}/${config.GITHUB_REPO}/actions/workflows/${config.GITHUB_WORKFLOW_FILE}/dispatches`,
      {
        method: "POST",
        headers: apiHeaders(installation.token),
        body: JSON.stringify({
          ref: config.GITHUB_REF,
          inputs: { watchdog: true },
        }),
      },
    );
  });
}

export async function handleScheduled(
  scheduledTimeMs,
  env,
  fetchImpl = fetch,
  authenticationTimeMs = Date.now(),
) {
  const clock = sydneyClock(scheduledTimeMs);
  const localTime =
    `${String(clock.year).padStart(4, "0")}-` +
    `${String(clock.month).padStart(2, "0")}-` +
    `${String(clock.day).padStart(2, "0")} ` +
    `${String(clock.hour).padStart(2, "0")}:` +
    `${String(clock.minute).padStart(2, "0")} Australia/Sydney`;

  if (!isDispatchWindow(scheduledTimeMs)) {
    console.log(`Watchdog idle outside the recovery window (${localTime}).`);
    return { dispatched: false, localTime };
  }

  const result = await dispatchGate(env, fetchImpl, authenticationTimeMs);
  console.log(
    `Watchdog dispatched the guarded prediction gate (${localTime}, run ${result.workflow_run_id || "accepted"}).`,
  );
  return { dispatched: true, localTime, result };
}

export default {
  async scheduled(controller, env, context) {
    context.waitUntil(handleScheduled(controller.scheduledTime, env));
  },
};
