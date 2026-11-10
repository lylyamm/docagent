# Deployment on Cloud Run

How DocAgent is deployed on Google Cloud. Every push to `main` that passes the CI is
deployed by [deploy.yml](../.github/workflows/deploy.yml) (see *Automatic deployment*);
the manual commands below do the same by hand. The resources were created by hand,
before Terraform (every resource as code). PowerShell syntax (a backtick continues the
line); in bash, use `\`.

Project `docagent-lilia`, region `europe-west9` (Paris). The architecture is in the
main [README](../README.md#deployment-on-google-cloud).

## Once: the resources

Created once, by hand (Terraform later):

| Resource | Name | Role |
| --- | --- | --- |
| Artifact Registry repository | `docagent` (Docker, europe-west9) | the images |
| Cloud Storage bucket | `docagent-lilia-index` | `index/` (passages + Qdrant vectors), `raw/` (corpus PDFs) |
| Secret | `mistral-api-key` | the translator's key, created in the console, never typed in a terminal |
| Secrets | `langfuse-public-key`, `langfuse-secret-key` | the tracing keys (Langfuse Cloud, EU), created in the console |
| Service account | `docagent-run` | the API: `roles/aiplatform.user` (project), `roles/storage.objectViewer` (bucket), `roles/secretmanager.secretAccessor` (the three secrets) |
| Service account | `docagent-front` | the front end: `roles/run.invoker` on the API only |
| Service account | `docagent-deployer` | GitHub Actions, see *Automatic deployment* |
| Workload Identity pool and provider | `github` / `docagent-repo` | accepts GitHub's tokens for `lylyamm/docagent` on `main` only |

Data, after building the index locally (see the main README):

```powershell
gcloud storage cp -r data/index gs://docagent-lilia-index/
gcloud storage cp -r data/raw gs://docagent-lilia-index/
```

## 1. Build the image

Cloud Build builds the [Dockerfile](../Dockerfile) next to the registry; only the files
not excluded by [.gcloudignore](../.gcloudignore) are uploaded (never `.env`). The tag
is the commit, so the running code is always known.

```powershell
gcloud builds submit --config cloudbuild.yaml --region europe-west9 --project docagent-lilia `
  --substitutions "_TAG=$(git rev-parse --short HEAD)"
```

Take the tag of the image built (not of the folder, which may have newer commits):

```powershell
$TAG = gcloud artifacts docker images list europe-west9-docker.pkg.dev/docagent-lilia/docagent/docagent `
  --include-tags --sort-by=~UPDATE_TIME --limit=1 --format="value(tags)"
```

## 2. Deploy the API

Settings in [api.env.yaml](api.env.yaml) (no secret in it); the Mistral key comes from
Secret Manager.

```powershell
gcloud run deploy docagent-api `
  --image europe-west9-docker.pkg.dev/docagent-lilia/docagent/docagent:$TAG `
  --region europe-west9 --project docagent-lilia `
  --service-account docagent-run@docagent-lilia.iam.gserviceaccount.com `
  --no-allow-unauthenticated `
  --port 8000 --cpu 2 --memory 4Gi `
  --min-instances 0 --max-instances 1 `
  --no-cpu-throttling --cpu-boost `
  --env-vars-file deploy/api.env.yaml `
  --set-secrets "MISTRAL_API_KEY=mistral-api-key:latest,LANGFUSE_PUBLIC_KEY=langfuse-public-key:latest,LANGFUSE_SECRET_KEY=langfuse-secret-key:latest"
```

| Option | Why |
| --- | --- |
| `--no-allow-unauthenticated` | private: a request without a Google identity token gets 403 |
| `--min-instances 0 --max-instances 1` | nothing to pay when idle, a cost that cannot run away |
| `--no-cpu-throttling` | translations and agent runs continue after the 202 response |
| `--cpu-boost` | more CPU while the container starts (torch, the e5 model) |

A model change is a settings change: edit `api.env.yaml`, run the same command with the
same image.

## 3. Deploy the front end

Same image, Streamlit instead of uvicorn.

```powershell
gcloud run services add-iam-policy-binding docagent-api --region europe-west9 --project docagent-lilia `
  --member "serviceAccount:docagent-front@docagent-lilia.iam.gserviceaccount.com" --role roles/run.invoker

gcloud run deploy docagent-front `
  --image europe-west9-docker.pkg.dev/docagent-lilia/docagent/docagent:$TAG `
  --region europe-west9 --project docagent-lilia `
  --service-account docagent-front@docagent-lilia.iam.gserviceaccount.com `
  --no-allow-unauthenticated `
  --port 8501 --cpu 1 --memory 1Gi `
  --min-instances 0 --max-instances 1 `
  --session-affinity --timeout 3600 `
  --command streamlit `
  --args "run,front/app.py,--server.port=8501,--server.address=0.0.0.0,--server.headless=true" `
  --set-env-vars DOCAGENT_API_URL=https://docagent-api-51066993529.europe-west9.run.app
```

On Cloud Run (`K_SERVICE` is set), the front end asks the metadata server for an
identity token for `DOCAGENT_API_URL` and sends it with every call. A new
`run.invoker` grant can take a few minutes to apply (403 meanwhile).

## 4. Open and check

```powershell
gcloud run services proxy docagent-front --region europe-west9 --project docagent-lilia --port 8501
# http://localhost:8501

gcloud run services proxy docagent-api --region europe-west9 --project docagent-lilia --port 8080
# http://localhost:8080/docs

gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=docagent-api" `
  --project docagent-lilia --limit 30 --freshness 30m --format "value(timestamp,httpRequest.status,textPayload)"
```

After a quiet period the first call waits for a cold start: about a minute before the
API has copied the data (`data downloaded`) and loaded the model (`search index ready`).

## Demo access (IAP)

IAP on `docagent-front`, access for the members of `docagent-demo@googlegroups.com`.

Once, in the console:

1. Google Auth Platform: audience *External*, home page `https://lylyamm.github.io/docagent/`,
   privacy policy `https://lylyamm.github.io/docagent/privacy.html`, authorized domain
   `lylyamm.github.io`, then *Publish app*.
2. Cloud Run, `docagent-front`, *Security*: enable IAP, *Custom OAuth*, *Auto generate
   credentials*, *Save*.

Access for the group:

```powershell
gcloud iap web add-iam-policy-binding --resource-type=cloud-run --service=docagent-front `
  --region=europe-west9 --project docagent-lilia `
  --member "group:docagent-demo@googlegroups.com" --role roles/iap.httpsResourceAccessor
```

Access denied page (only this field: `gcloud iap settings set` replaces every setting,
including the OAuth client):

```powershell
$token = gcloud auth print-access-token
$url = "https://iap.googleapis.com/v1/projects/51066993529/iap_web/cloud_run-europe-west9/services/docagent-front:iapSettings?updateMask=iapSettings.applicationSettings.accessDeniedPageSettings"
$body = '{"applicationSettings":{"accessDeniedPageSettings":{"accessDeniedPageUri":"https://lylyamm.github.io/docagent/"}}}'
Invoke-RestMethod -Method Patch -Uri $url -Headers @{Authorization = "Bearer $token"} -ContentType "application/json" -Body $body
```

Check:

```powershell
gcloud iap settings get --project docagent-lilia --resource-type=cloud-run --service=docagent-front --region=europe-west9
gcloud iap web get-iam-policy --resource-type=cloud-run --service=docagent-front `
  --region=europe-west9 --project docagent-lilia
```

## Automatic deployment (GitHub Actions)

[deploy.yml](../.github/workflows/deploy.yml) runs after a successful CI on a push to
`main` (or by hand, *Run workflow*): Cloud Build builds the image tagged with the commit,
the API then the front end get a new revision with the settings of this folder, then
`GET /health` is called with an identity token. Everything happens in the job's log.

**No key in GitHub: Workload Identity Federation.** Each run receives a token signed by
GitHub that says which repository and branch it runs for. Google's Security Token
Service checks the signature and the pool's condition (`lylyamm/docagent`, `refs/heads/main`),
then gives a one-hour token of `docagent-deployer`.

```powershell
gcloud iam workload-identity-pools create github --project docagent-lilia --location global --display-name "GitHub Actions"
gcloud iam workload-identity-pools providers create-oidc docagent-repo `
  --project docagent-lilia --location global --workload-identity-pool github `
  --issuer-uri "https://token.actions.githubusercontent.com" `
  --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" `
  --attribute-condition "assertion.repository == 'lylyamm/docagent' && assertion.ref == 'refs/heads/main'"
gcloud iam service-accounts add-iam-policy-binding docagent-deployer@docagent-lilia.iam.gserviceaccount.com `
  --project docagent-lilia --role roles/iam.workloadIdentityUser `
  --member "principalSet://iam.googleapis.com/projects/51066993529/locations/global/workloadIdentityPools/github/attribute.repository/lylyamm/docagent"
```

What `docagent-deployer` may do, and nothing else:

| Role | On | Why |
| --- | --- | --- |
| `roles/cloudbuild.builds.editor`, `roles/serviceusage.serviceUsageConsumer` | project | start a build and follow its status |
| `roles/storage.bucketViewer` | project | list the buckets: `gcloud builds submit` checks that the staging bucket belongs to the project (without it: "forbidden … serviceusage.services.use") |
| `roles/storage.objectAdmin`, `roles/storage.legacyBucketReader` | bucket `docagent-lilia_cloudbuild` | upload the code to build |
| `roles/run.developer` | project | new revisions of the services (not their access) |
| `roles/artifactregistry.reader` | repository `docagent` | `gcloud run deploy` resolves the image tag to its digest |
| `roles/iam.serviceAccountUser` | the Cloud Build account, `docagent-run`, `docagent-front` | run the build and the services as these accounts |
| `roles/run.invoker` | `docagent-api` | the `/health` check |

The deployment does not pass `--no-allow-unauthenticated`: access is set once, and
`run.developer` may not change it. The build is started with `--async` and its status
read every 20 s: streaming its log would need read access to the whole project
(Viewer), so the job prints a link to the log in the Cloud Build page instead.

**Old images.** The registry keeps the 3 most recent images and deletes the others
after a day ([registry-cleanup.json](registry-cleanup.json)):

```powershell
gcloud artifacts repositories set-cleanup-policies docagent --project docagent-lilia --location europe-west9 `
  --policy deploy/registry-cleanup.json --no-dry-run
```

## Tracing (Langfuse)

Every request is traced in Langfuse Cloud (EU region, free Hobby plan): one trace per
question, agent run or translation, one generation per LLM call (model, tokens, cost),
one span per node of the agent's graph, and the judge's verdicts as scores
(`judge_kept_ratio`, `judge_removed`, `answered`). Traces from Cloud Run carry the
environment `cloud-run`, those from a laptop `local`.

The two keys live in Secret Manager and reach the API as environment variables:

```powershell
foreach ($s in "langfuse-public-key", "langfuse-secret-key") {
  gcloud secrets add-iam-policy-binding $s --project docagent-lilia `
    --member "serviceAccount:docagent-run@docagent-lilia.iam.gserviceaccount.com" `
    --role roles/secretmanager.secretAccessor
}
```

Without the keys (tests, CI, a local run), tracing is off and nothing changes.

