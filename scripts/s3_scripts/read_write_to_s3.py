import boto3
import pandas as pd
from io import BytesIO, StringIO

s3 = boto3.client("s3")


def read_csv_from_s3(bucket: str, key: str, **read_csv_kwargs) -> pd.DataFrame:
    """
    Reads a CSV object from S3 into a pandas DataFrame.
    """
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    return pd.read_csv(BytesIO(body), **read_csv_kwargs)


def write_df_to_s3(df: pd.DataFrame, bucket: str, key: str, **to_csv_kwargs) -> None:
    """
    Writes a pandas DataFrame to S3 as a CSV object (no local file).
    """
    buf = StringIO()
    df.to_csv(buf, index=False, **to_csv_kwargs)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=buf.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )