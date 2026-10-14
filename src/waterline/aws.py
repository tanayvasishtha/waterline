"""AWS backends: DynamoDB store, S3 evidence, and the Lambda entrypoint.

Local mode never imports this; it stays cheap. Deployed mode calls the same
waterline.poller with AWS-backed stores, keeping one code path for reads.
"""

import json
import os
from typing import TypeVar

import boto3
from boto3.dynamodb.conditions import Key
from pydantic import BaseModel

from waterline.poller import poll_site
from waterline.settings import Settings

T = TypeVar("T", bound=BaseModel)

KIND_PREFIX = {"reading": "r", "trace": "t", "flag": "f"}


class DynamoStore:
    """Single-table store: pk = site#kind, sk = timestamp (or trace id)."""

    def __init__(self, table: str, region: str = "us-east-1") -> None:
        self.table = boto3.resource("dynamodb", region_name=region).Table(table)

    def append(self, item: BaseModel, kind: str) -> None:
        d = item.model_dump(mode="json")
        site = d.get("site_id", "_")
        sk = d.get("time") or d.get("opened_at") or d.get("finished_at") or d.get("trace_id")
        self.table.put_item(
            Item={
                "pk": f"{site}#{KIND_PREFIX[kind]}",
                "sk": str(sk),
                "kind": kind,
                "data": json.dumps(d),
            }
        )

    def read(self, model: type[T], kind: str, site_id: str) -> list[T]:
        resp = self.table.query(
            KeyConditionExpression=Key("pk").eq(f"{site_id}#{KIND_PREFIX[kind]}")
        )
        return [model.model_validate(json.loads(i["data"])) for i in resp.get("Items", [])]


def put_evidence(bucket: str, key: str, body: bytes, region: str = "us-east-1") -> None:
    boto3.client("s3", region_name=region).put_object(
        Bucket=bucket, Key=key, Body=body, ContentType="image/jpeg"
    )


def handler(event: dict[str, object], context: object) -> dict[str, object]:
    """EventBridge entry: poll every site in the event's site list (or all).

    Env: WATERLINE_DATA_DIR must be under /tmp (the only writable path),
    WATERLINE_DDB_TABLE names the item table, WATERLINE_BEDROCK_MODEL_ID is
    optional (empty keeps the deterministic policy), and
    WATERLINE_USGS_API_KEY is optional (anonymous USGS access is fine).
    """
    settings = Settings.from_env()
    raw_sites = event.get("sites")
    site_ids = list(raw_sites) if isinstance(raw_sites, list) else ["hancock", "in_overland"]
    table = os.environ.get("WATERLINE_DDB_TABLE", "waterline")
    region = os.environ.get("AWS_REGION", "us-east-1")
    store = DynamoStore(table, region)
    results = [vars(poll_site(s, settings.data_dir, persist=store.append)) for s in site_ids]
    return {"polled": len(results), "results": results}
