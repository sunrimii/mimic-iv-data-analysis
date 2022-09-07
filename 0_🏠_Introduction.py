import pandas as pd
import streamlit as st

st.markdown(
    """
### Data

This analysis uses the publicly available medical dataset MIMIC, which contains deidentified electronic medical records of thousands of adult patients in medical and surgical intensive care units and emergency department wards, to retrieve a sample of patients with schizophrenia and bipolar disorder from MIMIC and query various diagnostic information for each patient.

### Topics

Various statistical methods were used to verify whether patients with severe mental illness were hospitalized for ischemic or hemorrhagic stroke, and whether patients with severe mental illness were rehospitalized for digestive system disorders.

### Methods

The main methods used in the analysis were independence tests:

* **t test** for mean age
* **U test** for median age
* **KS test** for maximum absolute difference in age
* **chi square test** for gender, event and whether they were diagnosed with each covariate
* **Fisher's exact tests** for gender, event and whether they were diagnosed with each covariate

survival analysis:

* **Kaplan-Meier estimator** was used to estimate survival functions
* **log-rank test** was used to verify whether there were statistically significant differences between the two survival functions
* **proportional hazard model** was used to estimate the risk ratio of each risk factor

and time-delay analysis:

* **cross correlation** was used to find whether there was a time delay
between survival functions
"""
)
