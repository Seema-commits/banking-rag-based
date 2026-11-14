"""Create the owner alerts for the Cloud Run service: one command, safe to re-run.

Run in Cloud Shell, from the project folder:

    python3 ops/alerts/setup_alerts.py --email you@example.com

It creates (skipping anything that already exists):
  - an email notification channel for --email
  - log-based metrics that count the app's structured log events
  - alert policies that email you on: Groq key rejected, Groq rate limit / quota, chat errors,
    web search failing, slow answers, and daily usage nearing its limit
and checks that the project's billing account has a budget alert.

Use --dry-run to print what would be created without calling Google Cloud.
Only the Python standard library and the gcloud CLI are needed.
"""

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

PREFIX = "[Thistlewood] "
LOGGING = "https://logging.googleapis.com/v2"
MONITORING = "https://monitoring.googleapis.com/v3"
BUDGETS = "https://billingbudgets.googleapis.com/v1"


def counter(name, description, log_filter):
    return {"name": name, "description": description, "filter": log_filter}


def metrics(service):
    base = f'resource.type="cloud_run_revision" AND resource.labels.service_name="{service}"'
    chat = f'{base} AND jsonPayload.event="chat"'
    return [
        counter("owner_llm_auth_errors", "Answers that failed because Groq rejected the API key",
                f'{chat} AND jsonPayload.outcome="llm_auth_error"'),
        counter("owner_llm_rate_limited", "Answers that failed because of a Groq rate limit or quota",
                f'{chat} AND jsonPayload.outcome="llm_rate_limited"'),
        counter("owner_chat_errors", "Answers that failed for any reason",
                f'{chat} AND (jsonPayload.outcome="error" OR jsonPayload.outcome=~"^llm_")'),
        counter("owner_web_search_failures", "Web searches that failed (bad key, credits, timeout)",
                f'{base} AND jsonPayload.event="web_search_failed"'),
        counter("owner_usage_thresholds", "Daily usage reached 80% or 100% of a configured limit",
                f'{base} AND jsonPayload.event="usage_threshold"'),
        {   # same definition as ops/monitoring/chat_latency.yaml, so the dashboard can reuse it
            "name": "chat_latency",
            "description": "End-to-end chat latency in milliseconds",
            "filter": chat,
            "valueExtractor": "EXTRACT(jsonPayload.latency_ms)",
            "metricDescriptor": {"metricKind": "DELTA", "valueType": "DISTRIBUTION", "unit": "ms"},
            "bucketOptions": {"exponentialBuckets": {"numFiniteBuckets": 24, "growthFactor": 1.5, "scale": 20}},
        },
    ]


RUNBOOK = "https://github.com/Seema-commits/banking-rag-based/blob/main/docs/alerts.md"


def policies(latency_ms):
    def count_policy(title, metric, minutes, above, doc):
        return {
            "displayName": PREFIX + title,
            "combiner": "OR",
            "conditions": [{
                "displayName": title,
                "conditionThreshold": {
                    "filter": f'metric.type="logging.googleapis.com/user/{metric}" AND resource.type="cloud_run_revision"',
                    "aggregations": [{"alignmentPeriod": f"{minutes * 60}s", "perSeriesAligner": "ALIGN_SUM",
                                      "crossSeriesReducer": "REDUCE_SUM"}],
                    "comparison": "COMPARISON_GT",
                    "thresholdValue": above,
                    "duration": "0s",
                    "trigger": {"count": 1},
                },
            }],
            "documentation": {"mimeType": "text/markdown", "content": f"{doc}\n\nRunbook: {RUNBOOK}"},
            "alertStrategy": {"autoClose": "1800s"},
        }

    latency = {
        "displayName": PREFIX + "Slow answers",
        "combiner": "OR",
        "conditions": [{
            "displayName": f"95th percentile answer time above {latency_ms // 1000} s",
            "conditionThreshold": {
                "filter": 'metric.type="logging.googleapis.com/user/chat_latency" AND resource.type="cloud_run_revision"',
                "aggregations": [{"alignmentPeriod": "900s", "perSeriesAligner": "ALIGN_PERCENTILE_95",
                                  "crossSeriesReducer": "REDUCE_MAX"}],
                "comparison": "COMPARISON_GT",
                "thresholdValue": latency_ms,
                "duration": "0s",
                "trigger": {"count": 1},
            },
        }],
        "documentation": {"mimeType": "text/markdown", "content": (
            "Answers are slow. Check which step is slow: Logs Explorer, query "
            "`jsonPayload.event=\"chat\"`, compare grade_ms, web_search_ms, generate_ms, grounding_ms."
            f"\n\nRunbook: {RUNBOOK}")},
        "alertStrategy": {"autoClose": "1800s"},
    }
    return [
        count_policy("Groq API key rejected", "owner_llm_auth_errors", 5, 0,
                     "Groq rejected the API key, so the bot can't answer. Create a new key at "
                     "console.groq.com/keys and add it as a new version of the `groq-api-key` secret."),
        count_policy("Groq rate limit or quota reached", "owner_llm_rate_limited", 10, 0,
                     "Groq refused requests because of a rate limit or quota. Check usage at console.groq.com; "
                     "lower CHAT_REQUESTS_PER_DAY or upgrade the Groq plan."),
        count_policy("Chat errors", "owner_chat_errors", 10, 3,
                     "More than 3 questions failed in 10 minutes. Check the service logs for `outcome` and `error_type`."),
        count_policy("Web search failing", "owner_web_search_failures", 15, 1,
                     "Web searches are failing. Check the `reason` field: auth = Tavily key invalid, "
                     "quota = credits used up, timeout/network = Tavily unreachable."),
        count_policy("Daily usage near limit", "owner_usage_thresholds", 5, 0,
                     "A daily counter (questions, LLM tokens or web searches) reached 80% or 100% of its limit. "
                     "The log event shows which metric and how much was used."),
        latency,
    ]


class Api:
    def __init__(self, project, dry_run):
        self.project, self.dry_run = project, dry_run
        self.token = None if dry_run else gcloud("auth", "print-access-token")

    def call(self, method, url, body=None, quota_project=False):
        if self.dry_run:
            raise RuntimeError("no API calls in dry-run")
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        if quota_project:
            headers["x-goog-user-project"] = self.project
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise RuntimeError(f"{method} {url} -> {exc.code}: {exc.read().decode(errors='replace')[:500]}")


def gcloud(*args):
    result = subprocess.run(["gcloud", *args], capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"gcloud {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def ensure_metrics(api, items):
    for metric in items:
        url = f"{LOGGING}/projects/{api.project}/metrics"
        if api.dry_run:
            print(f"  would create log metric {metric['name']}")
            continue
        if api.call("GET", f"{url}/{metric['name']}") is not None:
            print(f"  log metric {metric['name']}: already exists")
        else:
            api.call("POST", url, metric)
            print(f"  log metric {metric['name']}: created")


def ensure_channel(api, email):
    if api.dry_run:
        print(f"  would create email channel for {email}")
        return "projects/PROJECT/notificationChannels/DRY_RUN"
    url = f"{MONITORING}/projects/{api.project}/notificationChannels"
    listing = api.call("GET", url + "?" + urllib.parse.urlencode({"filter": 'type="email"'})) or {}
    for channel in listing.get("notificationChannels", []):
        if channel.get("labels", {}).get("email_address", "").lower() == email.lower():
            print(f"  email channel for {email}: already exists")
            return channel["name"]
    created = api.call("POST", url, {"type": "email", "displayName": f"{PREFIX}owner ({email})",
                                     "labels": {"email_address": email}})
    print(f"  email channel for {email}: created")
    return created["name"]


def ensure_policies(api, items, channel):
    if api.dry_run:
        for policy in items:
            print(f"  would create alert policy {policy['displayName']}")
        return
    url = f"{MONITORING}/projects/{api.project}/alertPolicies"
    existing = {p["displayName"] for p in (api.call("GET", url + "?pageSize=200") or {}).get("alertPolicies", [])}
    for policy in items:
        if policy["displayName"] in existing:
            print(f"  alert {policy['displayName']}: already exists")
            continue
        api.call("POST", url, {**policy, "notificationChannels": [channel]})
        print(f"  alert {policy['displayName']}: created")


def check_budget(api):
    if api.dry_run:
        print("  would check the billing account's budgets")
        return
    account = gcloud("billing", "projects", "describe", api.project, "--format=value(billingAccountName)")
    if not account:
        print("  ! No billing account linked to this project.")
        return
    try:
        gcloud("services", "enable", "billingbudgets.googleapis.com")
        budgets = (api.call("GET", f"{BUDGETS}/{account}/budgets", quota_project=True) or {}).get("budgets", [])
    except (RuntimeError, SystemExit) as exc:
        print(f"  ! Could not read budgets ({str(exc)[:120]}). Check Billing > Budgets & alerts in the console.")
        return
    if not budgets:
        print("  ! No budget found. Create one: Billing > Budgets & alerts > Create budget (see docs/alerts.md).")
    for b in budgets:
        amount = b.get("amount", {}).get("specifiedAmount", {})
        print(f"  budget '{b.get('displayName', '?')}': {amount.get('units', '?')} {amount.get('currencyCode', '')}, "
              f"alerts at {[round(r.get('thresholdPercent', 0) * 100) for r in b.get('thresholdRules', [])]}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--email", required=True, help="where alert emails are sent")
    parser.add_argument("--service", default="thistlewood-assistant", help="Cloud Run service name")
    parser.add_argument("--project", help="Google Cloud project ID (default: gcloud config)")
    parser.add_argument("--slow-answer-seconds", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="print the plan without calling Google Cloud")
    args = parser.parse_args()

    project = args.project or ("PROJECT" if args.dry_run else gcloud("config", "get-value", "project"))
    print(f"Project: {project}   Service: {args.service}   Alerts to: {args.email}\n")
    api = Api(project, args.dry_run)
    if not args.dry_run:
        print("Enabling Cloud Monitoring and Logging APIs ...")
        gcloud("services", "enable", "monitoring.googleapis.com", "logging.googleapis.com")

    print("1/4 Log-based metrics")
    ensure_metrics(api, metrics(args.service))
    print("2/4 Email notification channel")
    channel = ensure_channel(api, args.email)
    print("3/4 Alert policies")
    ensure_policies(api, policies(args.slow_answer_seconds * 1000), channel)
    print("4/4 Budget alert")
    check_budget(api)
    print("\nDone. Alerts appear in Monitoring > Alerting; emails go to", args.email)
    print("Try it: docs/alerts.md > 'Send yourself a test alert'.")


if __name__ == "__main__":
    main()
