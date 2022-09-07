from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from utils.enum import Gender, StreamlitEnum


def read_month_counts() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_pickle("data/month_counts_psychosis.pkl"),
        pd.read_pickle("data/month_counts_digestive_disorders.pkl"),
    )


class DiseaseFilter1(StreamlitEnum):
    ORIGIN = "Psychosis"
    MINUS = "Psychosis - Digestive Disorders (control)"
    INTER = "Psychosis ⋂ Digestive Disorders (case)"  # is equivalent to Digestive Disorders ⋂ Psychosis


class DiseaseFilter2(StreamlitEnum):
    ORIGIN = "Digestive Disorders"
    MINUS = "Digestive Disorders - Psychosis"
    INTER = "Digestive Disorders ⋂ Psychosis"  # is equivalent to Psychosis ⋂ Digestive Disorders


def filter_disease(
    month_counts_1: pd.DataFrame, month_counts_2: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    disease_filter_1: DiseaseFilter1 = st.sidebar.radio("set", DiseaseFilter1.to_list())
    disease_filter_2: DiseaseFilter2 = st.sidebar.radio("set", DiseaseFilter2.to_list())

    month_counts_1.attrs["disease_filter"] = disease_filter_1
    month_counts_2.attrs["disease_filter"] = disease_filter_2

    match disease_filter_1:
        case DiseaseFilter1.ORIGIN:
            month_counts_1_ = month_counts_1
        case DiseaseFilter1.MINUS:
            month_counts_1_ = month_counts_1.query("~index.isin(@month_counts_2.index)")
        case DiseaseFilter1.INTER:
            month_counts_1_ = month_counts_1.query("index.isin(@month_counts_2.index)")
    match disease_filter_2:
        case DiseaseFilter2.ORIGIN:
            month_counts_2_ = month_counts_2
        case DiseaseFilter2.MINUS:
            month_counts_2_ = month_counts_2.query("~index.isin(@month_counts_1.index)")
        case DiseaseFilter2.INTER:
            month_counts_2_ = month_counts_2.query("index.isin(@month_counts_1.index)")

    if (
        disease_filter_1 == DiseaseFilter1.INTER
        and disease_filter_2 == DiseaseFilter2.INTER
    ):
        assert len(month_counts_1_) == len(month_counts_2_)

    return month_counts_1_, month_counts_2_


def filter_gender(
    month_counts_1: pd.DataFrame, month_counts_2: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gender_selected: Gender = st.sidebar.radio("gender", Gender.to_list())
    match gender_selected:
        case Gender.ALL:
            return month_counts_1, month_counts_2
        case Gender.MALE:
            return (
                month_counts_1.query("gender == 1"),
                month_counts_2.query("gender == 1"),
            )
        case Gender.FEMALE:
            return (
                month_counts_1.query("gender == 0"),
                month_counts_2.query("gender == 0"),
            )
        case _:
            raise AttributeError


def filter_age(
    month_counts_1: pd.DataFrame, month_counts_2: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    age_threshold: tuple[int, int] = st.sidebar.slider("age", 0, 110, (0, 65))
    month_counts_1 = month_counts_1.query(
        f"{age_threshold[0]} <= age <= {age_threshold[1]}"
    )
    month_counts_2 = month_counts_2.query(
        f"{age_threshold[0]} <= age <= {age_threshold[1]}"
    )
    return month_counts_1, month_counts_2


def make_corr_section(
    month_counts_1: pd.DataFrame, month_counts_2: pd.DataFrame
) -> pd.DataFrame:
    disease_filter_1 = month_counts_1.attrs["disease_filter"]
    disease_filter_2 = month_counts_2.attrs["disease_filter"]
    cnt_cols = [
        "jan_cnt",
        "feb_cnt",
        "mar_cnt",
        "apr_cnt",
        "may_cnt",
        "june_cnt",
        "july_cnt",
        "aug_cnt",
        "sept_cnt",
        "oct_cnt",
        "nov_cnt",
        "dec_cnt",
    ]
    corr_df = pd.DataFrame(
        {
            disease_filter_1: month_counts_1[cnt_cols].sum(),
            disease_filter_2: month_counts_2[cnt_cols].sum(),
        },
    )
    corr_df.index = pd.Index(range(-6, 6))
    corr_df.attrs["disease_filter_1"] = disease_filter_1
    corr_df.attrs["disease_filter_2"] = disease_filter_2

    def calc_corr(s1: pd.Series, s2: pd.Series) -> np.ndarray:
        s1 = (s1 - s1.mean()) / s1.std()
        s2 = (s2 - s2.mean()) / s2.std()
        return np.correlate(s1, s2, "same") / len(s1)

    corr_df["corr"] = calc_corr(corr_df[disease_filter_1], corr_df[disease_filter_2])
    return corr_df


def show_corr_section(corr_df: pd.DataFrame) -> None:
    fig, axs = plt.subplots(3, 1, sharex=True, constrained_layout=True)
    axs[0].plot(corr_df.iloc[:, 0])
    axs[1].plot(corr_df.iloc[:, 1])
    axs[2].plot(corr_df.iloc[:, 2])
    axs[0].set_title(corr_df.attrs["disease_filter_1"])
    axs[1].set_title(corr_df.attrs["disease_filter_2"])
    axs[2].set_title("Cross Correlation")
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    axs[1].set_xlabel("month")
    axs[2].set_xticks(range(-6, 6))
    axs[2].set_xlabel("lag")

    st.header("Cross Correlation")
    st.pyplot(fig)
    st.dataframe(corr_df, width=1000)
    st.markdown("---")


if __name__ == "__main__":
    month_counts_1, month_counts_2 = read_month_counts()
    month_counts_1, month_counts_2 = filter_disease(month_counts_1, month_counts_2)
    month_counts_1, month_counts_2 = filter_gender(month_counts_1, month_counts_2)
    month_counts_1, month_counts_2 = filter_age(month_counts_1, month_counts_2)
    show_corr_section(make_corr_section(month_counts_1, month_counts_2))
