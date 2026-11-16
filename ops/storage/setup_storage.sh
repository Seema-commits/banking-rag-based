#!/usr/bin/env bash
# Permanent storage for the Cloud Run service: one command, safe to re-run.
#
#   bash ops/storage/setup_storage.sh
#
# Creates (skipping anything that already exists):
#   - a Firestore database for the audit log (questions, answers, feedback), with a TTL policy
#     that deletes records after INTERACTION_RETENTION_DAYS (set in the app, default 90 days)
#   - a private Cloud Storage bucket for uploaded PDFs
#   - permissions for the app's service account
# and then points the Cloud Run service at them (STORAGE_BACKEND=firestore, DOCUMENTS_BUCKET).
#
# Options (environment variables): SERVICE (default thistlewood-assistant), REGION (us-central1),
# BUCKET (default <project>-documents), DRY_RUN=1 to print the commands without running them.
set -euo pipefail

SERVICE=${SERVICE:-thistlewood-assistant}
REGION=${REGION:-us-central1}
PROJECT=${PROJECT:-$(gcloud config get-value project 2>/dev/null)}
BUCKET=${BUCKET:-${PROJECT}-documents}

run() {
  echo "+ $*"
  if [ -z "${DRY_RUN:-}" ]; then "$@"; fi
}
exists() {  # true if the command succeeds (used to skip things that already exist)
  [ -z "${DRY_RUN:-}" ] && "$@" >/dev/null 2>&1
}

[ -n "$PROJECT" ] || { echo "No project set. Run: gcloud config set project YOUR_PROJECT_ID"; exit 1; }
echo "Project: $PROJECT   Service: $SERVICE   Region: $REGION   Bucket: gs://$BUCKET"
echo

echo "1/5 Enabling Firestore and Cloud Storage APIs"
run gcloud services enable firestore.googleapis.com storage.googleapis.com

echo "2/5 Firestore database (audit log)"
if exists gcloud firestore databases describe --database="(default)"; then
  echo "  already exists"
else
  run gcloud firestore databases create --database="(default)" --location="$REGION" --type=firestore-native
fi
echo "  auto-delete old records (TTL on the expire_at field)"
run gcloud firestore fields ttls update expire_at --collection-group=interactions --enable-ttl --async --quiet || \
  echo "  (TTL may already be enabled; check Firestore > Time-to-live in the console)"

echo "3/5 Cloud Storage bucket (uploaded PDFs)"
if exists gcloud storage buckets describe "gs://$BUCKET"; then
  echo "  already exists"
else
  run gcloud storage buckets create "gs://$BUCKET" --location="$REGION" \
    --uniform-bucket-level-access --public-access-prevention
fi

echo "4/5 Permissions for the app's service account"
NUMBER=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)' 2>/dev/null || echo PROJECT_NUMBER)
SA="${NUMBER}-compute@developer.gserviceaccount.com"
run gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$SA" \
  --role=roles/datastore.user --condition=None --quiet --format=none
run gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --member="serviceAccount:$SA" \
  --role=roles/storage.objectAdmin --quiet --format=none

echo "5/5 Point the Cloud Run service at the new storage"
run gcloud run services update "$SERVICE" --region "$REGION" \
  --update-env-vars "STORAGE_BACKEND=firestore,DOCUMENTS_BUCKET=$BUCKET"

echo
echo "Done. Questions, answers and feedback now go to Firestore; uploaded PDFs to gs://$BUCKET."
echo "Check: Admin > Insights should say 'storage: firestore' after a question is asked."
