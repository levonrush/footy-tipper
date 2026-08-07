import assert from "node:assert/strict";
import { generateKeyPairSync } from "node:crypto";
import { after, before, test } from "node:test";

import {
  createAppJwt,
  dispatchGate,
  handleScheduled,
  isDispatchWindow,
  sydneyClock,
} from "../src/index.js";

let privateKeyPem;
let publicKeySpki;
let originalConsoleLog;

before(() => {
  const pair = generateKeyPairSync("rsa", { modulusLength: 2048 });
  privateKeyPem = pair.privateKey.export({ type: "pkcs1", format: "pem" });
  publicKeySpki = pair.publicKey.export({ type: "spki", format: "der" });
  originalConsoleLog = console.log;
  console.log = () => {};
});

after(() => {
  console.log = originalConsoleLog;
});

function environment() {
  return {
    GITHUB_OWNER: "levonrush",
    GITHUB_REPO: "footy-tipper",
    GITHUB_WORKFLOW_FILE: "predict.yml",
    GITHUB_REF: "main",
    GITHUB_APP_ID: "12345",
    GITHUB_INSTALLATION_ID: "67890",
    GITHUB_APP_PRIVATE_KEY: privateKeyPem,
  };
}

function decodeBase64Url(value) {
  const padding = "=".repeat((4 - (value.length % 4)) % 4);
  return Buffer.from(
    value.replaceAll("-", "+").replaceAll("_", "/") + padding,
    "base64",
  );
}

test("Sydney recovery window handles AEST and AEDT", () => {
  const aest = Date.parse("2026-07-16T01:27:00Z");
  const aedt = Date.parse("2026-03-05T00:57:00Z");

  assert.equal(isDispatchWindow(aest), true);
  assert.equal(isDispatchWindow(aedt), true);
  assert.deepEqual(
    { hour: sydneyClock(aest).hour, minute: sydneyClock(aest).minute },
    { hour: 11, minute: 27 },
  );
  assert.deepEqual(
    { hour: sydneyClock(aedt).hour, minute: sydneyClock(aedt).minute },
    { hour: 11, minute: 57 },
  );
});

test("Sydney recovery window excludes wrong hours and minutes", () => {
  assert.equal(isDispatchWindow(Date.parse("2026-07-16T00:57:00Z")), false);
  assert.equal(isDispatchWindow(Date.parse("2026-07-16T05:27:00Z")), false);
  assert.equal(isDispatchWindow(Date.parse("2026-07-16T01:28:00Z")), false);
});

test("GitHub App JWT accepts GitHub PKCS1 keys and has a valid signature", async () => {
  const now = Date.parse("2026-08-06T03:00:00Z");
  const jwt = await createAppJwt("12345", privateKeyPem, now);
  const [header, payload, signature] = jwt.split(".");
  const claims = JSON.parse(decodeBase64Url(payload).toString("utf8"));

  assert.deepEqual(
    JSON.parse(decodeBase64Url(header).toString("utf8")),
    { alg: "RS256", typ: "JWT" },
  );
  assert.equal(claims.iss, "12345");
  assert.equal(claims.iat, Math.floor(now / 1000) - 60);
  assert.equal(claims.exp - claims.iat, 9 * 60);
  const verificationKey = await crypto.subtle.importKey(
    "spki",
    publicKeySpki,
    { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" },
    false,
    ["verify"],
  );
  assert.equal(
    await crypto.subtle.verify(
      "RSASSA-PKCS1-v1_5",
      verificationKey,
      decodeBase64Url(signature),
      new TextEncoder().encode(`${header}.${payload}`),
    ),
    true,
  );
});

test("dispatch requests a repository-restricted Actions token and guarded input", async () => {
  const requests = [];
  const fetchMock = async (url, options) => {
    requests.push({ url, options });
    if (url.includes("/access_tokens")) {
      return new Response(JSON.stringify({ token: "installation-token" }), {
        status: 201,
      });
    }
    return new Response(
      JSON.stringify({
        workflow_run_id: 42,
        html_url: "https://github.com/levonrush/footy-tipper/actions/runs/42",
      }),
      { status: 200 },
    );
  };

  const result = await dispatchGate(
    environment(),
    fetchMock,
    Date.parse("2026-08-06T03:00:00Z"),
  );

  assert.equal(result.workflow_run_id, 42);
  assert.equal(requests.length, 2);
  assert.deepEqual(JSON.parse(requests[0].options.body), {
    repositories: ["footy-tipper"],
    permissions: { actions: "write" },
  });
  assert.deepEqual(JSON.parse(requests[1].options.body), {
    ref: "main",
    inputs: { watchdog: true },
  });
  assert.equal(
    requests[1].url,
    "https://api.github.com/repos/levonrush/footy-tipper/actions/workflows/predict.yml/dispatches",
  );
  assert.equal(requests[1].options.headers.Authorization, "Bearer installation-token");
});

test("transient GitHub failures are retried", async () => {
  let tokenRequests = 0;
  let dispatchRequests = 0;
  const fetchMock = async (url) => {
    if (url.includes("/access_tokens")) {
      tokenRequests += 1;
      return new Response(JSON.stringify({ token: `token-${tokenRequests}` }), {
        status: 201,
      });
    }
    dispatchRequests += 1;
    if (dispatchRequests === 1) return new Response("temporary", { status: 503 });
    return new Response(JSON.stringify({ workflow_run_id: 43 }), { status: 200 });
  };

  const result = await dispatchGate(
    environment(),
    fetchMock,
    Date.parse("2026-08-06T03:00:00Z"),
  );

  assert.equal(result.workflow_run_id, 43);
  assert.equal(tokenRequests, 2);
  assert.equal(dispatchRequests, 2);
});

test("permanent GitHub failures are redacted and not retried", async () => {
  let requests = 0;
  const fetchMock = async () => {
    requests += 1;
    return new Response("super-secret-response-body", { status: 403 });
  };

  await assert.rejects(
    dispatchGate(
      environment(),
      fetchMock,
      Date.parse("2026-08-06T03:00:00Z"),
    ),
    (error) => {
      assert.match(error.message, /status 403/u);
      assert.doesNotMatch(error.message, /super-secret/u);
      return true;
    },
  );
  assert.equal(requests, 1);
});

test("scheduled handler does not authenticate or dispatch outside the window", async () => {
  let requests = 0;
  const result = await handleScheduled(
    Date.parse("2026-07-16T00:27:00Z"),
    environment(),
    async () => {
      requests += 1;
      throw new Error("fetch must not run");
    },
  );

  assert.equal(result.dispatched, false);
  assert.equal(requests, 0);
});
