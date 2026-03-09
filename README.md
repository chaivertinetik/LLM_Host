# LLM_Host
This repository contains the hosted Github CI/CD for LLM Erdo


The repo offers two functionalities :
1) Running the LLM Assitant script locally: installs micromamba directly from micro.mamba.pm and includes OS detection to download the correct version for macOS or Linux

and 

2) For CI/CD automated workflow: first creates a virutal ubuntu machine and then clones the code to the runner. Then creates the virtual environment using the environment file. It finally activate the llm-geo environment, installs any dependencies from requirements.txt and executes your LLM_Geo_Executable.py script.


The worflow automation triggers when: Someone opens a pull request against the main branch or You manually trigger it via the GitHub Actions UI using the "workflow_dispatch" event


Conncecting Vertex to hosted github: https://docs.google.com/document/d/1yaeu1pjWm-B51a8NmYapWe1_28ITkxCHbAruwEJ1gq8/edit?usp=sharing

## Cloud SQL / PostGIS support

This repo now supports Google Cloud SQL PostgreSQL / PostGIS-backed data locations in addition to ArcGIS REST layers.

### New runtime pieces
- `gcp_sql.py` adds Cloud SQL connector utilities and PostGIS ROI querying.
- `GET /cloudsql/geojson` returns GeoJSON directly from Cloud SQL and applies the active project ROI when `project_name` is supplied.
- `GET /cloudsql/describe` returns available columns, SRID, and estimated row count for a PostGIS table.
- `config/data_locations.yml` can now define `source: cloudsql` entries.

### Required environment variables
- `CLOUDSQL_INSTANCE_CONNECTION_NAME` — project:region:instance
- `CLOUDSQL_DB_NAME`
- `CLOUDSQL_DB_USER`
- `CLOUDSQL_DB_PASSWORD` — not needed when `CLOUDSQL_ENABLE_IAM_AUTH=1`
- `CLOUDSQL_ENABLE_IAM_AUTH` — optional, `1` to use IAM DB auth
- `APP_BASE_URL` — public base URL for this FastAPI service so generated data locations can point at `/cloudsql/geojson`

### Example YAML entry
```yaml
- label: "Cloud SQL Trees"
  source: "cloudsql"
  schema: "public"
  table: "tree_crowns"
  geom_column: "geom"
  columns: ["tree_id", "species", "health", "height"]
  where: "is_active = true"
  spatial_op: "intersects"
  aoifilter: true
```

When `aoifilter: true`, the generated data location is an internal ROI-aware GeoJSON URL rather than an ArcGIS URL, so the LLM can keep using the existing data-location workflow without needing ArcGIS tokens.
