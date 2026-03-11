import json
import os
import re
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode

from shapely.geometry import box, mapping, shape

try:
    from google.cloud.sql.connector import Connector
except Exception:  # pragma: no cover
    Connector = None

try:
    import pg8000
except Exception:  # pragma: no cover
    pg8000 = None

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALLOWED_SPATIAL_OPS = {"intersects", "within", "contains", "dwithin"}
DEFAULT_APP_BASE_URL = os.getenv(
    "APP_BASE_URL",
    os.getenv("SERVICE_URL", "https://llmgeo-dev-1042524106019.us-central1.run.app"),
).rstrip("/")


def _require_env(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if value is None or str(value).strip() == "":
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return str(value).strip()


@lru_cache(maxsize=1)
def _get_connector():
    if Connector is None:
        raise RuntimeError(
            "google-cloud-sql-connector is not installed. Add it to requirements.txt."
        )
    return Connector(refresh_strategy="LAZY")


@lru_cache(maxsize=1)
def _db_config() -> Dict[str, Any]:
    return {
        "instance_connection_name": _require_env("CLOUDSQL_INSTANCE_CONNECTION_NAME"),
        "db_user": _require_env("CLOUDSQL_DB_USER"),
        "db_pass": _require_env("CLOUDSQL_DB_PASSWORD", ""),
        "db_name": _require_env("CLOUDSQL_DB_NAME"),
        "use_iam_auth": os.getenv("CLOUDSQL_ENABLE_IAM_AUTH", "0").strip().lower() in {"1", "true", "yes"},
    }


@lru_cache(maxsize=1)
def _connect_kwargs() -> Dict[str, Any]:
    cfg = _db_config()
    kwargs: Dict[str, Any] = {
        "instance_connection_string": cfg["instance_connection_name"],
        "driver": "pg8000",
        "user": cfg["db_user"],
        "db": cfg["db_name"],
    }
    if cfg["use_iam_auth"]:
        kwargs["enable_iam_auth"] = True
    else:
        kwargs["password"] = cfg["db_pass"]
    return kwargs


def get_connection():
    if pg8000 is None:
        raise RuntimeError("pg8000 is not installed. Add it to requirements.txt.")
    connector = _get_connector()
    return connector.connect(**_connect_kwargs())


def validate_identifier(name: str, kind: str = "identifier") -> str:
    name = str(name or "").strip()
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid {kind}: {name!r}")
    return name


def quote_ident(name: str) -> str:
    return f'"{validate_identifier(name)}"'


def normalize_columns(columns: Optional[Iterable[str]]) -> List[str]:
    out: List[str] = []
    for col in columns or []:
        col = str(col).strip()
        if not col:
            continue
        out.append(validate_identifier(col, "column"))
    return out


def resolve_source_spec(source: Dict[str, Any]) -> Dict[str, Any]:
    schema = validate_identifier(str(source.get("schema") or "public"), "schema")
    table = validate_identifier(str(source.get("table") or ""), "table")
    geom_column = validate_identifier(str(source.get("geom_column") or "geom"), "geometry column")
    spatial_op = str(source.get("spatial_op") or "intersects").strip().lower()
    if spatial_op not in _ALLOWED_SPATIAL_OPS:
        raise ValueError(f"Unsupported spatial_op: {spatial_op}")
    limit = int(source.get("limit") or os.getenv("CLOUDSQL_QUERY_LIMIT", "5000"))
    limit = max(1, min(limit, 50000))
    srid = source.get("srid")
    srid = int(srid) if srid not in (None, "") else None
    distance_m = source.get("distance_m")
    distance_m = float(distance_m) if distance_m not in (None, "") else None
    where = str(source.get("where") or "").strip()
    columns = normalize_columns(source.get("columns") or source.get("fields"))
    order_by = normalize_columns(source.get("order_by"))
    return {
        "schema": schema,
        "table": table,
        "geom_column": geom_column,
        "columns": columns,
        "where": where,
        "srid": srid,
        "limit": limit,
        "spatial_op": spatial_op,
        "distance_m": distance_m,
        "order_by": order_by,
    }


@lru_cache(maxsize=128)
def get_available_columns(schema: str, table: str, geom_column: str) -> List[str]:
    sql = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, (schema, table))
        cols = [row[0] for row in cur.fetchall()]
    return [c for c in cols if c != geom_column]


@lru_cache(maxsize=128)
def get_table_srid(schema: str, table: str, geom_column: str) -> int:
    sql = f"SELECT COALESCE(NULLIF(Find_SRID(%s, %s, %s), 0), 4326)"
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, (schema, table, geom_column))
        row = cur.fetchone()
    return int((row or [4326])[0] or 4326)


@lru_cache(maxsize=128)
def get_estimated_row_count(schema: str, table: str) -> Optional[int]:
    sql = """
        SELECT reltuples::bigint
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = %s AND c.relname = %s
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, (schema, table))
        row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else None


@lru_cache(maxsize=128)
def describe_source(schema: str, table: str, geom_column: str) -> Dict[str, Any]:
    cols = get_available_columns(schema, table, geom_column)
    srid = get_table_srid(schema, table, geom_column)
    est = get_estimated_row_count(schema, table)
    return {
        "schema": schema,
        "table": table,
        "geom_column": geom_column,
        "srid": srid,
        "columns": cols,
        "estimated_row_count": est,
    }


def _build_spatial_filter(geom_sql: str, spatial_op: str, table_srid: int, distance_m: Optional[float]):
    if spatial_op == "intersects":
        return f"ST_Intersects({geom_sql}, ST_Transform(ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), %s))", [table_srid]
    if spatial_op == "within":
        return f"ST_Within({geom_sql}, ST_Transform(ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), %s))", [table_srid]
    if spatial_op == "contains":
        return f"ST_Contains({geom_sql}, ST_Transform(ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), %s))", [table_srid]
    if spatial_op == "dwithin":
        distance_m = 0.0 if distance_m is None else float(distance_m)
        return (
            f"ST_DWithin(ST_Transform({geom_sql}, 4326)::geography, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)::geography, %s)",
            [distance_m],
        )
    raise ValueError(f"Unsupported spatial_op: {spatial_op}")


def _feature_signature(feature: Dict[str, Any]) -> str:
    return json.dumps(feature, sort_keys=True, default=str)


def _chunk_count_from_area(bounds: Tuple[float, float, float, float], target_chunk_area: float) -> int:
    minx, miny, maxx, maxy = bounds
    width = max(maxx - minx, 0.0)
    height = max(maxy - miny, 0.0)
    area = width * height
    if area <= 0:
        return 1
    return max(1, int(math.ceil(area / max(target_chunk_area, 1e-9))))


def _iter_aoi_chunks(aoi_geojson: Dict[str, Any], *, max_chunks: int, target_chunk_area_deg2: float):
    geom = shape(aoi_geojson)
    if geom.is_empty:
        return []

    minx, miny, maxx, maxy = geom.bounds
    if minx == maxx or miny == maxy:
        return [mapping(geom)]

    approx_chunks = _chunk_count_from_area((minx, miny, maxx, maxy), target_chunk_area_deg2)
    chunk_count = max(1, min(max_chunks, approx_chunks))
    grid_n = max(1, int(math.ceil(math.sqrt(chunk_count))))

    dx = (maxx - minx) / grid_n
    dy = (maxy - miny) / grid_n
    chunks = []
    for row in range(grid_n):
        for col in range(grid_n):
            cell = box(minx + col * dx, miny + row * dy, minx + (col + 1) * dx, miny + (row + 1) * dy)
            clipped = geom.intersection(cell)
            if not clipped.is_empty:
                chunks.append(mapping(clipped))
    return chunks or [mapping(geom)]


def _fetch_geojson_single(source: Dict[str, Any], aoi_geojson: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    spec = resolve_source_spec(source)
    schema = spec["schema"]
    table = spec["table"]
    geom_column = spec["geom_column"]
    table_info = describe_source(schema, table, geom_column)
    table_srid = spec["srid"] or int(table_info["srid"])

    columns = spec["columns"] or list(table_info["columns"])
    columns = [c for c in columns if c != geom_column]

    select_parts = [f"{quote_ident(c)}" for c in columns]
    select_clause = ", ".join(select_parts) + (", " if select_parts else "")
    geom_sql = quote_ident(geom_column)
    table_sql = f"{quote_ident(schema)}.{quote_ident(table)}"

    sql = f"SELECT {select_clause}ST_AsGeoJSON(ST_Transform({geom_sql}, 4326)) AS __geometry_geojson FROM {table_sql}"
    where_parts: List[str] = []
    params: List[Any] = []

    if spec["where"] and spec["where"] != "1=1":
        where_parts.append(f"({spec['where']})")

    if aoi_geojson:
        spatial_sql, extra_params = _build_spatial_filter(
            geom_sql,
            spec["spatial_op"],
            table_srid,
            spec["distance_m"],
        )
        where_parts.append(spatial_sql)
        params.append(json.dumps(aoi_geojson))
        params.extend(extra_params)

    if where_parts:
        sql += " WHERE " + " AND ".join(where_parts)

    if spec["order_by"]:
        sql += " ORDER BY " + ", ".join(quote_ident(c) for c in spec["order_by"])
    sql += " LIMIT %s"
    params.append(spec["limit"])

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()

    features = []
    for row in rows:
        props = {col: row[idx] for idx, col in enumerate(columns)}
        geom_json = row[len(columns)]
        geometry = json.loads(geom_json) if geom_json else None
        features.append({"type": "Feature", "geometry": geometry, "properties": props})

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            **table_info,
            "limit": spec["limit"],
            "spatial_op": spec["spatial_op"],
            "distance_m": spec["distance_m"],
            "roi_applied": bool(aoi_geojson),
        },
    }


def fetch_geojson(source: Dict[str, Any], aoi_geojson: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    spec = resolve_source_spec(source)

    chunking_enabled = str(source.get("chunking", os.getenv("CLOUDSQL_ROI_CHUNKING", "1"))).strip().lower() in {"1", "true", "yes"}
    max_chunks = max(1, min(int(source.get("max_chunks") or os.getenv("CLOUDSQL_MAX_CHUNKS", "16")), 64))
    max_workers = max(1, min(int(source.get("max_workers") or os.getenv("CLOUDSQL_MAX_WORKERS", str(min(8, max(2, (os.cpu_count() or 2)))))), 16))
    target_chunk_area_deg2 = float(source.get("target_chunk_area_deg2") or os.getenv("CLOUDSQL_TARGET_CHUNK_AREA_DEG2", "0.01"))

    if not aoi_geojson or not chunking_enabled:
        return _fetch_geojson_single(spec, aoi_geojson=aoi_geojson)

    chunks = _iter_aoi_chunks(
        aoi_geojson,
        max_chunks=max_chunks,
        target_chunk_area_deg2=target_chunk_area_deg2,
    )

    if len(chunks) <= 1:
        result = _fetch_geojson_single(spec, aoi_geojson=aoi_geojson)
        result.setdefault("metadata", {})["chunking"] = {
            "enabled": False,
            "chunks_requested": 1,
            "chunks_used": 1,
            "max_workers": 1,
        }
        return result

    merged_features: List[Dict[str, Any]] = []
    seen = set()

    with ThreadPoolExecutor(max_workers=min(max_workers, len(chunks))) as executor:
        futures = {executor.submit(_fetch_geojson_single, spec, chunk): idx for idx, chunk in enumerate(chunks)}
        first_meta: Optional[Dict[str, Any]] = None
        for future in as_completed(futures):
            result = future.result()
            if first_meta is None:
                first_meta = dict(result.get("metadata") or {})
            for feature in result.get("features", []):
                sig = _feature_signature(feature)
                if sig in seen:
                    continue
                seen.add(sig)
                merged_features.append(feature)

    metadata = first_meta or describe_source(spec["schema"], spec["table"], spec["geom_column"])
    metadata.update({
        "limit": spec["limit"],
        "spatial_op": spec["spatial_op"],
        "distance_m": spec["distance_m"],
        "roi_applied": True,
        "chunking": {
            "enabled": True,
            "chunks_requested": len(chunks),
            "chunks_used": len(chunks),
            "max_workers": min(max_workers, len(chunks)),
            "target_chunk_area_deg2": target_chunk_area_deg2,
            "deduplicated_features": len(merged_features),
        },
    })

    return {
        "type": "FeatureCollection",
        "features": merged_features,
        "metadata": metadata,
    }

def build_geojson_url(source: Dict[str, Any], project_name: Optional[str] = None, app_base_url: Optional[str] = None) -> str:
    spec = resolve_source_spec(source)
    base = (app_base_url or DEFAULT_APP_BASE_URL).rstrip("/")
    params: Dict[str, Any] = {
        "schema": spec["schema"],
        "table": spec["table"],
        "geom_column": spec["geom_column"],
        "limit": spec["limit"],
        "spatial_op": spec["spatial_op"],
    }
    if project_name:
        params["project_name"] = project_name
    if spec["columns"]:
        params["columns"] = ",".join(spec["columns"])
    if spec["where"]:
        params["where"] = spec["where"]
    if spec["srid"]:
        params["srid"] = spec["srid"]
    if spec["distance_m"] is not None:
        params["distance_m"] = spec["distance_m"]
    if spec["order_by"]:
        params["order_by"] = ",".join(spec["order_by"])
    params["chunking"] = os.getenv("CLOUDSQL_ROI_CHUNKING", "1")
    params["max_workers"] = os.getenv("CLOUDSQL_MAX_WORKERS", str(min(8, max(2, (os.cpu_count() or 2)))))
    params["max_chunks"] = os.getenv("CLOUDSQL_MAX_CHUNKS", "16")
    return f"{base}/cloudsql/geojson?{urlencode(params)}"
