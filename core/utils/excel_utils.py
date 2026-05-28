import pandas as pd


_COLUMN_ALIASES = {
    "Source": [
        "Source",
        "source",
        "\u539f\u6587",
        "\u6e90\u6587",
        "\u6e90\u6587\u672c",
        "\u539f\u59cb\u6587\u672c",
        "\u539f\u53e5",
        "\u539f\u5b57\u5e55",
    ],
    "Translation": [
        "Translation",
        "translation",
        "\u8bd1\u6587",
        "\u7ffb\u8bd1",
        "\u76ee\u6807\u6587\u672c",
        "\u7ffb\u8bd1\u6587\u672c",
        "\u8bd1\u5b57\u5e55",
        "\u4e2d\u6587\u5b57\u5e55",
    ],
}


def normalize_dataframe_columns(df: pd.DataFrame, required_columns=None) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]

    rename_map = {}
    for canonical_name, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalized.columns:
                rename_map[alias] = canonical_name
                break

    normalized = normalized.rename(columns=rename_map)

    if required_columns:
        missing = [column for column in required_columns if column not in normalized.columns]
        if missing:
            raise KeyError(
                f"Missing required columns {missing}. Available columns: {list(normalized.columns)}"
            )

    return normalized


def read_excel_with_aliases(path: str, required_columns=None, **kwargs) -> pd.DataFrame:
    df = pd.read_excel(path, **kwargs)
    return normalize_dataframe_columns(df, required_columns=required_columns)
