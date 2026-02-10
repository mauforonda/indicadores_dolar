#!/usr/bin/env python3

import argparse
import pandas as pd

from upload import upload_dataset


REPO = "mauforonda/dolares/refs/heads/main"
FN_DIFERENCIAL = "dolar_diferencia_referencial_binance_"
FN_REFERENCIAL = "dolar_referencial_"
TIPOS = ("sell", "buy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera series referenciales y su diferencia contra Binance "
            "para compra y venta."
        )
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Sube los datos a Supabase (por defecto solo guarda CSV).",
    )
    return parser.parse_args()


def load_referencial(tipo: str) -> pd.DataFrame:
    return pd.read_csv(
        f"https://raw.githubusercontent.com/{REPO}/{tipo}_oficial.csv",
        parse_dates=["timestamp"],
    )


def load_binance(tipo: str) -> pd.DataFrame:
    return pd.read_csv(
        f"https://raw.githubusercontent.com/{REPO}/{tipo}.csv",
        parse_dates=["timestamp"],
        usecols=["timestamp", "vwap"],
    )


def compute_residual(
    referencial: pd.DataFrame, binance: pd.DataFrame
) -> pd.DataFrame:
    # Ensure both inputs are sorted for merge_asof semantics.
    df_asof = (
        referencial[["timestamp", "value"]]
        .sort_values("timestamp")
        .rename(columns={"timestamp": "timestamp_df", "value": "value_df"})
    )

    binance_asof = (
        binance[["timestamp", "vwap"]]
        .sort_values("timestamp")
        .rename(columns={"vwap": "value_binance"})
    )

    binance_asof["timestamp_naive"] = binance_asof["timestamp"].dt.tz_localize(None)

    residual = pd.merge_asof(
        binance_asof,
        df_asof,
        left_on="timestamp_naive",
        right_on="timestamp_df",
        direction="backward",
    )

    result = residual[["timestamp", "value_binance", "value_df"]].copy()
    result["value"] = (result["value_binance"] - result["value_df"]).round(2)
    return result[["timestamp", "value"]].dropna()


def save_outputs(tipo: str, referencial: pd.DataFrame, residual: pd.DataFrame) -> None:
    referencial.to_csv(f"{FN_REFERENCIAL}{tipo}.csv", index=False)
    residual.to_csv(
        f"{FN_DIFERENCIAL}{tipo}.csv",
        float_format="%.2f",
        index=False,
    )


def maybe_upload(tipo: str, referencial: pd.DataFrame, residual: pd.DataFrame) -> None:
    upload_dataset(f"{FN_REFERENCIAL}{tipo}", referencial, ["timestamp"])
    upload_dataset(f"{FN_DIFERENCIAL}{tipo}", residual, ["timestamp"])


def main() -> None:
    args = parse_args()

    for tipo in TIPOS:
        referencial = load_referencial(tipo)
        binance = load_binance(tipo)
        residual = compute_residual(referencial, binance)
        save_outputs(tipo, referencial, residual)

        if args.upload:
            maybe_upload(tipo, referencial, residual)


if __name__ == "__main__":
    main()
