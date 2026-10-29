# Deployment on Cloud Run

The commands used to deploy DocAgent on Google Cloud, by hand, until they move to
GitHub Actions (automatic deployment) and Terraform (every resource as code).
PowerShell syntax (a backtick continues the line); in bash, use `\`.

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
