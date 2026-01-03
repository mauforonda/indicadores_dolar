#!/usr/bin/env python3

import pandas as pd
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter
from supabase import create_client
import os

SB_URL = os.environ["SUPABASE_URL"]
SB_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]


def on_transactions(df):
    amount, price = [
        df.pivot_table(
            index="timestamp", columns="advertiser_userno", values=c, aggfunc="last"
        )
        for c in ["tradablequantity", "price"]
    ]
    delta_amount = amount.diff()
    weights = (-delta_amount).clip(lower=0)
    vwap = (weights * price).sum(axis=1) / weights.sum(axis=1)
    demand = weights.sum(axis=1)
    return vwap, demand


def on_advs(df):
    vwap = pd.Series(
        {
            t: np.average(g.price, weights=g.tradablequantity)
            for t, g in df.groupby("timestamp")
        }
    )
    supply = df.groupby("timestamp").tradablequantity.sum()
    return vwap, supply


kg = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "andreschirinos/p2p-bob-exchange",
    "advice.parquet",
)
supabase = create_client(SB_URL, SB_KEY)

for trade_type in ["BUY", "SELL"]:
    tables = {"BUY": "dolar_sell", "SELL": "dolar_buy"}
    table = tables[trade_type]

    df = kg[(kg.asset == "USDT") & (kg.tradetype == trade_type)][
        ["advertiser_userno", "timestamp", "price", "tradablequantity"]
    ].copy()

    vwap_on_transactions, demand = on_transactions(df)
    vwap_on_advs, supply = on_advs(df)
    data = (
        pd.concat([vwap_on_transactions, vwap_on_advs, demand, supply], axis=1)
        .reset_index()
        .dropna()
    )
    data.columns = [
        "timestamp",
        "vwap_sale",
        "vwap_advs",
        "demand",
        "supply",
    ]
    data.timestamp = data.timestamp.dt.tz_localize("UTC")
    data.timestamp = data.timestamp.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    data.to_csv(f"{table}.csv", index=False)
    try:
        supabase.table(table).upsert(
            data.to_dict(orient="records"), on_conflict="timestamp"
        ).execute()
    except Exception as e:
        print(f"Error syncing with Supabase for table {table}: {e}")
