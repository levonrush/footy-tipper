import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import vm from "node:vm";

const SOURCE = fs.readFileSync(
  new URL("../src/Code.js", import.meta.url),
  "utf8",
);

function response(status, body = "") {
  return {
    getResponseCode: () => status,
    getContentText: () => body,
  };
}

function formatSydney(instant) {
  const parts = new Intl.DateTimeFormat("en-AU", {
    timeZone: "Australia/Sydney",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  }).formatToParts(instant);
  const values = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return `${values.year}-${values.month}-${values.day}T${values.hour}:${values.minute}`;
}

function loadWatchdog({
  fetch = () => response(204),
  initialProperties = {},
  triggers = [],
  now = "2026-06-04T01:22:00Z",
} = {}) {
  const properties = new Map(Object.entries(initialProperties));
  const fetchCalls = [];
  const sleeps = [];
  const deletedTriggers = [];
  const createdTriggers = [];
  const logs = [];
  let currentNow = new Date(now);

  const scriptProperties = {
    getProperty: (key) => properties.get(key) ?? null,
    setProperty: (key, value) => {
      properties.set(key, String(value));
      return scriptProperties;
    },
    deleteProperty: (key) => {
      properties.delete(key);
      return scriptProperties;
    },
  };

  class WatchdogDate extends Date {
    constructor(value) {
      super(value === undefined ? currentNow : value);
    }
  }

  const sandbox = {
    Date: WatchdogDate,
    JSON,
    Math,
    Number,
    Object,
    String,
    encodeURIComponent,
    Error,
    console: {
      log: (...values) => logs.push(values.map(String).join(" ")),
    },
    Utilities: {
      formatDate: (instant, timeZone, pattern) => {
        assert.equal(timeZone, "Australia/Sydney");
        assert.equal(pattern, "yyyy-MM-dd'T'HH:mm");
        return formatSydney(instant);
      },
      sleep: (milliseconds) => sleeps.push(milliseconds),
    },
    UrlFetchApp: {
      fetch: (url, options) => {
        fetchCalls.push({ url, options });
        return fetch(url, options, fetchCalls.length);
      },
    },
    PropertiesService: {
      getScriptProperties: () => scriptProperties,
    },
    LockService: {
      getScriptLock: () => ({
        tryLock: () => true,
        releaseLock: () => {},
      }),
    },
    ScriptApp: {
      getProjectTriggers: () => triggers,
      deleteTrigger: (trigger) => deletedTriggers.push(trigger),
      newTrigger: (handler) => ({
        timeBased: () => ({
          everyMinutes: (minutes) => ({
            create: () => {
              const trigger = {
                handler,
                minutes,
                getHandlerFunction: () => handler,
                getUniqueId: () => "new-trigger-id",
              };
              createdTriggers.push(trigger);
              return trigger;
            },
          }),
        }),
      }),
    },
  };
  vm.createContext(sandbox);
  vm.runInContext(SOURCE, sandbox, { filename: "Code.js" });
  return {
    sandbox,
    properties,
    fetchCalls,
    sleeps,
    deletedTriggers,
    createdTriggers,
    logs,
    setNow: (value) => {
      currentNow = new Date(value);
    },
  };
}

test("calculates identical first slots in AEST and AEDT", () => {
  const runtime = loadWatchdog();
  const aest = runtime.sandbox.recoverySlot_(
    new Date("2026-06-04T01:22:00Z"),
  );
  const aedt = runtime.sandbox.recoverySlot_(
    new Date("2026-12-03T00:22:00Z"),
  );
  assert.deepEqual({ ...aest }, { index: 0, key: "2026-06-04:0" });
  assert.deepEqual({ ...aedt }, { index: 0, key: "2026-12-03:0" });
});

test("enforces recovery boundaries and the final slot", () => {
  const runtime = loadWatchdog();
  assert.equal(
    runtime.sandbox.recoverySlot_(new Date("2026-06-04T01:21:00Z")),
    null,
  );
  assert.deepEqual(
    {
      ...runtime.sandbox.recoverySlot_(
        new Date("2026-06-04T04:52:00Z"),
      ),
    },
    { index: 7, key: "2026-06-04:7" },
  );
  assert.equal(
    runtime.sandbox.recoverySlot_(new Date("2026-06-04T05:00:00Z")),
    null,
  );
});

test("dispatches only once per successful recovery slot", () => {
  const runtime = loadWatchdog({
    initialProperties: { GITHUB_TOKEN: "slot-token" },
  });
  assert.equal(runtime.sandbox.watchdogTick(), true);
  assert.equal(runtime.sandbox.watchdogTick(), false);
  assert.equal(runtime.fetchCalls.length, 1);
  assert.equal(runtime.properties.get("LAST_SUCCESSFUL_SLOT"), "2026-06-04:0");
});

test("retries transient responses and records state only after success", () => {
  const statuses = [500, 429, 204];
  const runtime = loadWatchdog({
    initialProperties: { GITHUB_TOKEN: "retry-token" },
    fetch: () => response(statuses.shift()),
  });
  assert.equal(runtime.sandbox.watchdogTick(), true);
  assert.deepEqual(runtime.sleeps, [500, 2000]);
  assert.equal(runtime.fetchCalls.length, 3);
  assert.equal(runtime.properties.get("LAST_SUCCESSFUL_SLOT"), "2026-06-04:0");
});

test("permanent errors expose status only and do not mutate slot state", () => {
  const secret = "github_pat_secret-value";
  const body = '{"secret":"server-response-secret"}';
  const runtime = loadWatchdog({
    initialProperties: { GITHUB_TOKEN: secret },
    fetch: () => response(401, body),
  });
  assert.throws(
    () => runtime.sandbox.watchdogTick(),
    /GitHub request failed with HTTP 401\./,
  );
  assert.equal(runtime.properties.has("LAST_SUCCESSFUL_SLOT"), false);
  const visible = runtime.logs.join("\n");
  assert.equal(visible.includes(secret), false);
  assert.equal(visible.includes(body), false);
});

test("dispatch payload is restricted to the guarded gate", () => {
  const runtime = loadWatchdog({
    initialProperties: { GITHUB_TOKEN: "payload-token" },
  });
  runtime.sandbox.probeWatchdog();
  assert.equal(runtime.fetchCalls.length, 1);
  const call = runtime.fetchCalls[0];
  assert.equal(
    call.url,
    "https://api.github.com/repos/levonrush/footy-tipper/actions/workflows/predict.yml/dispatches",
  );
  assert.equal(call.options.method, "post");
  assert.deepEqual(JSON.parse(call.options.payload), {
    ref: "main",
    inputs: { watchdog: true },
  });
  assert.equal(call.options.headers.Authorization, "Bearer payload-token");
  assert.equal(call.options.headers["X-GitHub-Api-Version"], "2026-03-10");
});

test("installation validates actor, probes, and replaces only watchdog triggers", () => {
  const watchdogTrigger = {
    getHandlerFunction: () => "watchdogTick",
  };
  const unrelatedTrigger = {
    getHandlerFunction: () => "otherFunction",
  };
  const runtime = loadWatchdog({
    initialProperties: { GITHUB_TOKEN: "install-token" },
    triggers: [watchdogTrigger, unrelatedTrigger],
    fetch: (url) => {
      if (url.endsWith("/user")) {
        return response(200, '{"login":"levonrush"}');
      }
      return response(url.endsWith("/dispatches") ? 204 : 200, "{}");
    },
  });
  const result = runtime.sandbox.installWatchdog();
  assert.equal(result.actor, "levonrush");
  assert.equal(result.probeAccepted, true);
  assert.deepEqual(runtime.deletedTriggers, [watchdogTrigger]);
  assert.equal(runtime.createdTriggers.length, 1);
  assert.equal(runtime.createdTriggers[0].handler, "watchdogTick");
  assert.equal(runtime.createdTriggers[0].minutes, 5);
  assert.equal(runtime.fetchCalls.length, 3);
});

test("installation rejects a token for a different actor", () => {
  const runtime = loadWatchdog({
    initialProperties: { GITHUB_TOKEN: "wrong-actor-token" },
    fetch: () => response(200, '{"login":"someone-else"}'),
  });
  assert.throws(
    () => runtime.sandbox.installWatchdog(),
    /token actor does not match/,
  );
  assert.equal(runtime.createdTriggers.length, 0);
});

test("missing token fails before making an external request", () => {
  const runtime = loadWatchdog();
  assert.throws(
    () => runtime.sandbox.installWatchdog(),
    /GITHUB_TOKEN is missing/,
  );
  assert.equal(runtime.fetchCalls.length, 0);
});
