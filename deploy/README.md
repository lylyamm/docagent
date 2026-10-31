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
| Service account | `docagent-run` | the API: `roles/aiplatform.user` (project), `roles/storage.objectViewer` (bucket), `roles/secretmanager.secretAccessor` (secret) |
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
  --set-secrets MISTRAL_API_KEY=mistral-api-key:latest
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
| `roles/cloudbuild.builds.editor`, `roles/serviceusage.serviceUsageConsumer` | project | start a build |
| `roles/storage.objectAdmin`, `roles/storage.legacyBucketReader` | bucket `docagent-lilia_cloudbuild` | upload the code to build |
| `roles/run.developer` | project | new revisions of the services (not their access) |
| `roles/iam.serviceAccountUser` | the Cloud Build account, `docagent-run`, `docagent-front` | run the build and the services as these accounts |
| `roles/run.invoker` | `docagent-api` | the `/health` check |

The deployment does not pass `--no-allow-unauthenticated`: access is set once, and
`run.developer` may not change it. The build log is not streamed in the job
(`--suppress-logs`, which would need read access to the whole project): it is in the
Cloud Build page of the console.

**Old images.** The registry keeps the 3 most recent images and deletes the others
after a day ([registry-cleanup.json](registry-cleanup.json)):

```powershell
gcloud artifacts repositories set-cleanup-policies docagent --project docagent-lilia --location europe-west9 `
  --policy deploy/registry-cleanup.json --no-dry-run
```
