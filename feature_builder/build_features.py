import os
import argparse
from datetime import datetime
import sys
import psycopg2
import pandas as pd
from psycopg2.extras import execute_values

def get_connection():
    try:
        return psycopg2.connect(
            host=os.getenv("PG_HOST", "postgres"),
            port=os.getenv("PG_PORT", "5432"),
            dbname=os.getenv("PG_DB", "trading_db"),
            user=os.getenv("PG_USER", "root"),
            password=os.getenv("PG_PASSWORD", "root"),
        )
    except Exception as e:
        print("Error conectando a Postgres:", str(e))
        sys.exit(1)

def ensure_table():
    sql = """
    CREATE SCHEMA IF NOT EXISTS analytics;

    CREATE TABLE IF NOT EXISTS analytics.daily_features (
        date DATE NOT NULL,
        ticker TEXT NOT NULL,

        year INT,
        month INT,
        day_of_week INT,

        open DOUBLE PRECISION,
        high DOUBLE PRECISION,
        low DOUBLE PRECISION,
        close DOUBLE PRECISION,
        volume BIGINT,

        return_close_open DOUBLE PRECISION,
        return_prev_close DOUBLE PRECISION,
        volatility_5d DOUBLE PRECISION,

        close_lag1 DOUBLE PRECISION,

        run_id TEXT,
        ingested_at_utc TIMESTAMP,

        PRIMARY KEY (date, ticker)
    );
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()
    cur.close()
    conn.close()

    print("Tabla analytics.daily_features verificada")

def load_raw(ticker, start_date=None, end_date=None):
    conn = get_connection()

    query = """
        SELECT
            date,
            ticker,
            open,
            high,
            low,
            close,
            volume
        FROM raw.prices_daily
        WHERE ticker = %s
    """

    params = [ticker]

    if start_date:
        query += " AND date >= %s"
        params.append(start_date)

    if end_date:
        query += " AND date <= %s"
        params.append(end_date)

    query += " ORDER BY date ASC;"

    df = pd.read_sql(query, conn, params=params)
    conn.close()

    print(f"Raw cargado: {len(df)} filas")
    return df

def build_features(df_raw, ticker, run_id):

    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek

    df["return_close_open"] = (df["close"] - df["open"]) / df["open"]

    df["close_lag1"] = df["close"].shift(1)
    df["return_prev_close"] = df["close"] / df["close_lag1"] - 1

    df["volatility_5d"] = df["return_prev_close"].rolling(5).std()

    df["ticker"] = ticker
    df["run_id"] = run_id
    df["ingested_at_utc"] = datetime.utcnow()

    cols = [
        "date",
        "ticker",
        "year",
        "month",
        "day_of_week",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return_close_open",
        "return_prev_close",
        "volatility_5d",
        "close_lag1",
        "run_id",
        "ingested_at_utc",
    ]

    df_final = df[cols].dropna().reset_index(drop=True)

    print(f"Features creadas: {len(df_final)} filas")
    print("Desde:", df_final["date"].min(), "Hasta:", df_final["date"].max())

    return df_final


def delete_existing(ticker, start_date=None, end_date=None):
    conn = get_connection()
    cur = conn.cursor()

    query = "DELETE FROM analytics.daily_features WHERE ticker = %s"
    params = [ticker]

    if start_date:
        query += " AND date >= %s"
        params.append(start_date)

    if end_date:
        query += " AND date <= %s"
        params.append(end_date)

    cur.execute(query, params)
    deleted = cur.rowcount

    conn.commit()
    cur.close()
    conn.close()

    print("Filas eliminadas por overwrite:", deleted)

def insert_features(df, overwrite):

    if df.empty:
        print("No hay datos para insertar")
        return

    conn = get_connection()
    cur = conn.cursor()

    insert_sql = """
    INSERT INTO analytics.daily_features (
        date,
        ticker,
        year,
        month,
        day_of_week,
        open,
        high,
        low,
        close,
        volume,
        return_close_open,
        return_prev_close,
        volatility_5d,
        close_lag1,
        run_id,
        ingested_at_utc
    ) VALUES %s
    """

    if not overwrite:
        insert_sql += " ON CONFLICT (date, ticker) DO NOTHING"

    records = [
        tuple(row)
        for row in df[
            [
                "date",
                "ticker",
                "year",
                "month",
                "day_of_week",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "return_close_open",
                "return_prev_close",
                "volatility_5d",
                "close_lag1",
                "run_id",
                "ingested_at_utc",
            ]
        ].values
    ]

    execute_values(cur, insert_sql, records)
    conn.commit()
    cur.close()
    conn.close()

    print("Filas insertadas:", len(records))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "by-date-range"], required=True)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--overwrite", choices=["true", "false"], default="false")

    args = parser.parse_args()

    overwrite = args.overwrite == "true"

    print("===================================")
    print("FEATURE BUILDER - EJECUCIÓN")
    print("Ticker:", args.ticker)
    print("Modo:", args.mode)
    print("Run ID:", args.run_id)
    print("Overwrite:", overwrite)
    print("===================================")

    ensure_table()

    if args.mode == "full":
        df_raw = load_raw(args.ticker)
    else:
        if not args.start_date or not args.end_date:
            raise ValueError("by-date-range requiere start-date y end-date")
        df_raw = load_raw(args.ticker, args.start_date, args.end_date)

    df_feat = build_features(df_raw, args.ticker, args.run_id)

    if overwrite:
        delete_existing(args.ticker, args.start_date, args.end_date)

    insert_features(df_feat, overwrite)

    print("FEATURE BUILDER FINALIZADO CORRECTAMENTE")


if __name__ == "__main__":
    main()
