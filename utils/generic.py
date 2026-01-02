import numpy as np
import pandas as pd


def convert_types(df):
    for col in df.columns:
        if col in [
            "first_active_day",
            "last_active_day",
            "overall_first_active_day",
            "overall_last_active_day",
            "latest_active_day",
            "new_player_cutoff_date",
            "summary_date",
        ]:
            df[col.upper()] = pd.to_datetime(df[col.upper()], format="%Y-%m-%d")
        elif col in [
            "registration_timestamp",
            "conversion_timestamp",
            "game_start_timestamp_utc",
            "game_start_timestamp",
            "visit_start_timestamp_utc",
            "visit_end_timestamp_utc",
        ]:
            df[col.upper()] = pd.to_datetime(
                df[col.upper()], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce"
            )
        elif col in [
            "bet_amt_eur",
            "total_bet_amt",
            "freespins_bet_amt_eur",
            "suggested_bet_brand",
            "overall_bet_amt",
            "latest_total_bet_amt",
            "theoretical_hold_pct",
            "theoretical_win_pct",
            "bet_min_amt",
            "bet_max_amt",
        ]:
            df[col.upper()] = pd.to_numeric(df[col.upper()], downcast="float")
        elif col in [
            "player_id",
            "bet_qty",
            "raw_bet_qty",
            "raw_freespins_bet_qty",
            "number_of_visits",
            "total_spins",
            "apds",
            "freespins_bet_qty",
            "suggested_bet_default",
            "overall_number_of_visits",
            "overall_total_spins",
            "spin_rank",
            "bet_rank",
            "visits_rank",
            "latest_total_spins",
            "mobile_configered_yn",
            "desktop_configered_yn",
            "quick_seat_yn",
            "desktop_spins",
            "mobile_spins",
            "total_players",
            "total_mobile_games",
            "total_desktop_games",
            "jackpot_yn",
            "blackjack_yn",
            "roulette_yn",
            "live_yn",
            "embedded_yn",
            "mini_yn",
            "no_variants_in_game_group_and_jackpot",
            "lmt_direct_game_launch_yn",
            "wager_exclude_game_yn",
        ]:
            df[col.upper()] = pd.to_numeric(df[col.upper()], downcast="integer")

    return df


SCHEMA_MAP = dict(
    (k.lower(), v)
    for k, v in {
        "PLAYER_ID": np.int32,
        "FRONT_END_CD": "string",
        "SRC_BRAND_CD": "string",
        "SRC_BRAND_COUNTRY": "string",
        "LOGIN_NAME_TXT": "string",
        "SUMMARY_DATE": "datetime64[ms]",
        "CASIA_GAME_DESC": "string",
        "VISIT_START_TIMESTAMP_UTC": "datetime64[ms]",
        "BET_QTY": np.int32,
        "RAW_BET_QTY": np.int32,
        "ORIG_BET_QTY": np.int32,
        "LAST_ACTIVE_DAY": "string",
        "FREESPINS_BET_QTY": np.float32,
    }.items()
)
