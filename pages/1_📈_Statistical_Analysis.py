from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st
from more_itertools import chunked
from scipy.stats import (
    chi2_contingency,
    fisher_exact,
    ks_2samp,
    mannwhitneyu,
    ttest_ind,
)
from utils.display import format_small_values, highlight_small_p, to_percentage
from utils.enum import Gender


def read_design_matrix() -> pd.DataFrame:
    topic_to_pickle_fname = {
        "Psychosis & Ischemic Stroke": "design_matrix_psychosis_ischemic_stroke",
        "Psychosis & Hemorrhagic Stroke": "design_matrix_psychosis_hemorrhagic_stroke",
        "Digestive Disorders & Psychosis": "design_matrix_digestive_disorders_psychosis",
    }
    if "topic_index" not in st.session_state:
        st.session_state["topic_index"] = 0

    def on_change() -> None:
        topic_selected = st.session_state["topic-radio"]
        st.session_state["topic_index"] = list(topic_to_pickle_fname).index(
            topic_selected
        )

    topic_selected: str = st.sidebar.radio(
        "topic",
        options=topic_to_pickle_fname,
        index=st.session_state["topic_index"],
        key="topic-radio",  # to identify widgets on session state
        on_change=on_change,
    )
    design_matrix = pd.read_pickle(f"data/{topic_to_pickle_fname[topic_selected]}.pkl")

    # for slider usage
    design_matrix.attrs["max_duration"] = int(design_matrix["T"].max())

    return design_matrix


def filter_gender(design_matrix: pd.DataFrame) -> pd.DataFrame:
    if "gender_index" not in st.session_state:
        st.session_state["gender_index"] = 0

    def on_change() -> None:
        gender_selected = st.session_state["gender-radio"]
        st.session_state["gender_index"] = Gender.to_list().index(gender_selected)

    gender_selected: Gender = st.sidebar.radio(
        "gender",
        options=Gender.to_list(),
        index=st.session_state["gender_index"],
        key="gender-radio",  # to identify widgets on session state
        on_change=on_change,
    )
    design_matrix.attrs["gender_selected"] = gender_selected
    match gender_selected:
        case Gender.ALL:
            return design_matrix
        case Gender.MALE:
            return design_matrix.query("gender == 1")
        case Gender.FEMALE:
            return design_matrix.query("gender == 0")
        case _:
            raise AttributeError


def filter_age(design_matrix: pd.DataFrame) -> pd.DataFrame:
    if "age_threshold" not in st.session_state:
        st.session_state["age_threshold"] = 18, 50

    def on_change() -> None:
        st.session_state["age_threshold"] = st.session_state["age-slider"]

    age_threshold: tuple[int, int] = st.sidebar.slider(
        "age",
        min_value=0,
        max_value=110,
        value=st.session_state["age_threshold"],
        key="age-slider",
        on_change=on_change,
    )
    return design_matrix.query(f"{age_threshold[0]} <= age <= {age_threshold[1]}")


def crop_event(design_matrix: pd.DataFrame) -> pd.DataFrame:
    max_duration = design_matrix.attrs["max_duration"]
    lower, upper = st.sidebar.slider(
        "duration",
        min_value=0,
        max_value=max_duration,
        value=(0, max_duration),
        step=90,
    )

    # assemble thresholds of duration to event column
    if not (0 <= lower < upper <= max_duration):
        st.error("Please select the correct threshold for duration.")
        st.stop()
    if lower == 0 and upper == max_duration:
        event_col = "E"
    elif lower == 0 and upper > 0:
        event_col = f"E{upper}"
    else:
        event_col = f"E{lower}-{upper}"
    design_matrix.attrs["event_col"] = event_col

    # E is original event column. If threshold isn't the default, add additional event
    # column such as E90 or E91-180.
    if event_col != "E":
        design_matrix[event_col] = False
        design_matrix.loc[
            (design_matrix["E"] == True)
            & (lower <= design_matrix["T"])
            & (design_matrix["T"] <= upper),
            event_col,
        ] = True

    return design_matrix


@dataclass
class StatSubSection:
    """stat_section is composed of multiple StatSubSections."""

    subheader: str
    crosstab: pd.DataFrame
    stat_result: pd.DataFrame


def make_stat_section(
    design_matrix: pd.DataFrame,
) -> list[StatSubSection]:
    subsections: list[StatSubSection] = []

    predictor_col = design_matrix.attrs["predictor_col"]
    case = design_matrix.query(predictor_col)
    control = design_matrix.query(f"~{predictor_col}")

    # age
    crosstab = (
        design_matrix.groupby(predictor_col)["age"]
        .agg(("mean", "std"))
        .transpose()
        .rename(columns={True: "case", False: "control"})
    )
    t_res = ttest_ind(case["age"], control["age"])
    ks_res = ks_2samp(case["age"], control["age"])
    u_res = mannwhitneyu(case["age"], control["age"])
    stat_results = pd.DataFrame(
        [
            [t_res.statistic, t_res.pvalue],
            [u_res.statistic, u_res.pvalue],
            [ks_res.statistic, ks_res.pvalue],
        ],
        index=["t test", "U test", "KS test"],
        columns=["stat", "p"],
    )
    subsection = StatSubSection("Age", crosstab, stat_results)
    subsections.append(subsection)

    def make_catgorical_stat_results(crosstab: pd.DataFrame) -> pd.DataFrame:
        chi2_res = chi2_contingency(crosstab, correction=False)
        fe_res = fisher_exact(crosstab)
        return pd.DataFrame(
            [
                [chi2_res[0], chi2_res[1]],
                [fe_res[0], fe_res[1]],
            ],
            index=["chi2 test", "Fisher exact test"],
            columns=["stat", "p"],
        )

    # gender
    if design_matrix.attrs["gender_selected"] == Gender.ALL:
        crosstab = pd.crosstab(
            design_matrix["gender"], design_matrix[predictor_col]
        ).rename(
            index={1: "male", 0: "female"}, columns={True: "case", False: "control"}
        )
        stat_result = make_catgorical_stat_results(crosstab)
        subsection = StatSubSection("Gender", crosstab, stat_result)
        subsections.append(subsection)

    # event
    event_col = design_matrix.attrs["event_col"]
    crosstab = pd.crosstab(
        design_matrix[event_col], design_matrix[predictor_col]
    ).rename(
        index={True: "true", False: "false"},
        columns={True: "case", False: "control"},
    )
    stat_result = make_catgorical_stat_results(crosstab)
    subsection = StatSubSection("Event", crosstab, stat_result)
    subsections.append(subsection)

    # covariate
    cols = design_matrix.attrs["with_covariate_cols"]
    for col in cols:
        subheader = col.replace("_", " ").title()
        crosstab = pd.crosstab(design_matrix[col], design_matrix[predictor_col]).rename(
            index={True: "with", False: "without"},
            columns={True: "case", False: "control"},
        )
        stat_result = make_catgorical_stat_results(crosstab)
        subsection = StatSubSection(subheader, crosstab, stat_result)
        subsections.append(subsection)

    return subsections


def show_stat_section(subsections: list[StatSubSection]) -> None:
    st.header("Independence Tests")
    percentage = st.checkbox("percentage")
    for two_subsections in chunked(subsections, 2):
        for stcol, subsection in zip(st.columns(2), two_subsections):
            stcol.subheader(subsection.subheader)

            # show crosstab
            if subsection.subheader != "Age" and percentage:
                stcol.dataframe(to_percentage(subsection.crosstab))
            else:
                stcol.dataframe(subsection.crosstab)

            # show stat_result
            stcol.dataframe(
                subsection.stat_result.style.pipe(format_small_values).pipe(
                    highlight_small_p
                )
            )
    st.markdown("---")


if __name__ == "__main__":
    design_matrix = (
        read_design_matrix().pipe(filter_gender).pipe(filter_age).pipe(crop_event)
    )
    show_stat_section(make_stat_section(design_matrix))
