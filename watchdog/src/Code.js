/**
 * Independent delivery watchdog for Footy Tipper.
 *
 * The only secret is the repository-scoped fine-grained GitHub token stored in
 * the GITHUB_TOKEN Script Property. This script never logs that token or any
 * GitHub response body.
 */

var WATCHDOG_CONFIG = Object.freeze({
  owner: "levonrush",
  repo: "footy-tipper",
  workflow: "predict.yml",
  ref: "main",
  actor: "levonrush",
  timeZone: "Australia/Sydney",
  firstSlotMinute: 11 * 60 + 22,
  endMinuteExclusive: 15 * 60,
  slotMinutes: 30,
  handler: "watchdogTick",
  tokenProperty: "GITHUB_TOKEN",
  lastSlotProperty: "LAST_SUCCESSFUL_SLOT",
  apiVersion: "2026-03-10",
  userAgent: "footy-tipper-apps-script-watchdog",
});

var WATCHDOG_RETRY_DELAYS_MS = Object.freeze([500, 2000]);

/**
 * Validate the credential, send one guarded probe, and install the clock.
 *
 * Run this function manually once after GITHUB_TOKEN has been added in
 * Project Settings -> Script Properties. It is safe to run again: only
 * triggers owned by watchdogTick are replaced.
 */
function installWatchdog() {
  var token = requiredToken_();
  var actor = authenticatedActor_(token);
  if (actor !== WATCHDOG_CONFIG.actor) {
    throw new Error(
      "GitHub token actor does not match the configured watchdog actor."
    );
  }

  assertWorkflowAvailable_(token);
  dispatchGate_(token);

  var triggers = ScriptApp.getProjectTriggers();
  for (var index = 0; index < triggers.length; index += 1) {
    if (triggers[index].getHandlerFunction() === WATCHDOG_CONFIG.handler) {
      ScriptApp.deleteTrigger(triggers[index]);
    }
  }

  var trigger = ScriptApp.newTrigger(WATCHDOG_CONFIG.handler)
    .timeBased()
    .everyMinutes(5)
    .create();

  console.log(
    "Watchdog installed for actor %s with trigger %s.",
    actor,
    trigger.getUniqueId()
  );
  return {
    actor: actor,
    handler: WATCHDOG_CONFIG.handler,
    triggerId: trigger.getUniqueId(),
    probeAccepted: true,
  };
}

/**
 * Remove the watchdog clock without touching GitHub scheduling or the token.
 */
function uninstallWatchdog() {
  var removed = 0;
  var triggers = ScriptApp.getProjectTriggers();
  for (var index = 0; index < triggers.length; index += 1) {
    if (triggers[index].getHandlerFunction() === WATCHDOG_CONFIG.handler) {
      ScriptApp.deleteTrigger(triggers[index]);
      removed += 1;
    }
  }
  console.log("Removed %s watchdog trigger(s).", removed);
  return removed;
}

/**
 * Send a guarded gate request immediately without changing slot state.
 */
function probeWatchdog() {
  dispatchGate_(requiredToken_());
  console.log("Guarded watchdog probe accepted by GitHub.");
  return true;
}

/**
 * Return non-secret installation state for operator verification.
 */
function watchdogStatus() {
  var triggerCount = 0;
  var triggers = ScriptApp.getProjectTriggers();
  for (var index = 0; index < triggers.length; index += 1) {
    if (triggers[index].getHandlerFunction() === WATCHDOG_CONFIG.handler) {
      triggerCount += 1;
    }
  }
  var result = {
    configured:
      requiredScriptProperties_().getProperty(
        WATCHDOG_CONFIG.tokenProperty
      ) !== null,
    triggerCount: triggerCount,
    lastSuccessfulSlot:
      requiredScriptProperties_().getProperty(
        WATCHDOG_CONFIG.lastSlotProperty
      ) || null,
  };
  console.log(
    "Watchdog status: configured=%s triggerCount=%s lastSlot=%s",
    result.configured,
    result.triggerCount,
    result.lastSuccessfulSlot || "none"
  );
  return result;
}

/**
 * Five-minute trigger entrypoint. Only one successful dispatch is recorded for
 * each 30-minute Sydney recovery slot.
 */
function watchdogTick() {
  var lock = LockService.getScriptLock();
  if (!lock.tryLock(1000)) {
    console.log("Watchdog tick skipped because another execution holds the lock.");
    return false;
  }

  try {
    var slot = recoverySlot_(new Date());
    if (slot === null) {
      return false;
    }

    var properties = requiredScriptProperties_();
    if (
      properties.getProperty(WATCHDOG_CONFIG.lastSlotProperty) === slot.key
    ) {
      return false;
    }

    dispatchGate_(requiredToken_());
    properties.setProperty(WATCHDOG_CONFIG.lastSlotProperty, slot.key);
    console.log("Watchdog dispatched slot %s.", slot.key);
    return true;
  } finally {
    lock.releaseLock();
  }
}

/**
 * Map an instant to one of the eight Sydney recovery slots.
 */
function recoverySlot_(instant) {
  var local = Utilities.formatDate(
    instant,
    WATCHDOG_CONFIG.timeZone,
    "yyyy-MM-dd'T'HH:mm"
  );
  var match = local.match(/^(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2})$/);
  if (match === null) {
    throw new Error("Unable to calculate the Australia/Sydney recovery slot.");
  }

  var minuteOfDay = Number(match[2]) * 60 + Number(match[3]);
  if (
    minuteOfDay < WATCHDOG_CONFIG.firstSlotMinute ||
    minuteOfDay >= WATCHDOG_CONFIG.endMinuteExclusive
  ) {
    return null;
  }

  var index = Math.floor(
    (minuteOfDay - WATCHDOG_CONFIG.firstSlotMinute) /
      WATCHDOG_CONFIG.slotMinutes
  );
  return {
    index: index,
    key: match[1] + ":" + index,
  };
}

function requiredToken_() {
  var token = requiredScriptProperties_().getProperty(
    WATCHDOG_CONFIG.tokenProperty
  );
  if (token === null || token.trim() === "") {
    throw new Error(
      "GITHUB_TOKEN is missing from Apps Script Project Settings."
    );
  }
  return token.trim();
}

function requiredScriptProperties_() {
  return PropertiesService.getScriptProperties();
}

function authenticatedActor_(token) {
  var response = githubRequest_("GET", "/user", token, null);
  var body;
  try {
    body = JSON.parse(response.getContentText());
  } catch (error) {
    throw new Error("GitHub returned an unreadable actor response.");
  }
  if (!body || typeof body.login !== "string" || body.login === "") {
    throw new Error("GitHub actor response did not contain a login.");
  }
  return body.login;
}

function assertWorkflowAvailable_(token) {
  githubRequest_(
    "GET",
    "/repos/" +
      encodeURIComponent(WATCHDOG_CONFIG.owner) +
      "/" +
      encodeURIComponent(WATCHDOG_CONFIG.repo) +
      "/actions/workflows/" +
      encodeURIComponent(WATCHDOG_CONFIG.workflow),
    token,
    null
  );
}

function dispatchGate_(token) {
  githubRequest_(
    "POST",
    "/repos/" +
      encodeURIComponent(WATCHDOG_CONFIG.owner) +
      "/" +
      encodeURIComponent(WATCHDOG_CONFIG.repo) +
      "/actions/workflows/" +
      encodeURIComponent(WATCHDOG_CONFIG.workflow) +
      "/dispatches",
    token,
    {
      ref: WATCHDOG_CONFIG.ref,
      inputs: { watchdog: true },
    }
  );
}

function githubRequest_(method, path, token, payload) {
  var url = "https://api.github.com" + path;
  var options = {
    method: method.toLowerCase(),
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: "Bearer " + token,
      "User-Agent": WATCHDOG_CONFIG.userAgent,
      "X-GitHub-Api-Version": WATCHDOG_CONFIG.apiVersion,
    },
    muteHttpExceptions: true,
  };
  if (payload !== null) {
    options.contentType = "application/json";
    options.payload = JSON.stringify(payload);
  }

  for (
    var attempt = 0;
    attempt <= WATCHDOG_RETRY_DELAYS_MS.length;
    attempt += 1
  ) {
    var response;
    try {
      response = UrlFetchApp.fetch(url, options);
    } catch (error) {
      if (attempt < WATCHDOG_RETRY_DELAYS_MS.length) {
        Utilities.sleep(WATCHDOG_RETRY_DELAYS_MS[attempt]);
        continue;
      }
      throw new Error(
        "GitHub request failed after transient network retries."
      );
    }

    var status = response.getResponseCode();
    if (status === 200 || status === 204) {
      return response;
    }
    if (
      (status === 408 ||
        status === 409 ||
        status === 429 ||
        (status >= 500 && status <= 599)) &&
      attempt < WATCHDOG_RETRY_DELAYS_MS.length
    ) {
      Utilities.sleep(WATCHDOG_RETRY_DELAYS_MS[attempt]);
      continue;
    }
    throw new Error("GitHub request failed with HTTP " + status + ".");
  }

  throw new Error("GitHub request failed without a response.");
}
