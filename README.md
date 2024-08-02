# DrivenData - DengAI: Predicting Disease Spread Competition

## Background
Dengue fever is a significant global health concern, especially in tropical and subtropical regions where the Aedes mosquito, the primary vector for the disease, thrives. The accurate prediction of dengue cases can assist in early warning systems, enabling health authorities to implement timely measures such as vector control and public health advisories. In this notebook, we present an approach for dengue spread prediction, developed for submission in a [DrivenData data science competition](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/page/80/). Utilizing the provided dataset, which includes weather conditions (e.g., temperature, humidity, rainfall) and vegetation cover measured by the Normalized Difference Vegetation Index (NDVI), the objective is to forecast the weekly number of dengue cases in two cities: San Juan, Puerto Rico, and Iquitos, Peru.

_This submission placed in the 39th position on the leaderboard (teofc111), ranking within the top 1% of [submissions](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/leaderboard/)._

## Approach
__Auto-regression with XGBoost:__
In this notebook, an autoregressive model is implemented using XGBoost, making use of lagged features as well as lagged target values for training.

__Multi-step Forecasting Treatment:__
Given the multi-step forecasting nature of the problem, special techniques were applied to adapt XGBoost for predicting multiple time points ahead, to compensate for the lack of native multi-step forecasting support. Specifically, a multi-step prediction algorithm is used to make sequential predictions at every timestep, simultaneously updating the subsequent timesteps' lagged target values with the newly predicted target value from the current timestep.

__Hyperparameter Tuning and Model Construction:__
The XGBoost model hyperparameters and lags to adopt were selected via cross-validation. As an XGBoost model is trained under the assumption of perfect knowledge of lagged target values, as the correct lagged target values are supplied for every timestep as training data. This training objective thus naturally conflicts with the multi-step forecasting scenario, due to uncertainties associated with lagged target used for later timesteps. In this notebook, we account for these effects (elaborated below) when selecting the appropriate model parameters.

## Dataset
The dataset is freely available at [DrivenData](#https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/). The data description is reproduced below for reference.

| Feature                             | Type | Source | Description                                                                                     |
|-------------------------------------|------|--------|-------------------------------------------------------------------------------------------------|
| city                                |  obj    |    NA    | City abbreviations: sj for San Juan and iq for Iquitos                                          |
| week_start_date                     |  obj    |    NA    | Date given in yyyy-mm-dd format                                                                 |
| station_max_temp_c                  |  float    |  NOAA's GHCN      | Maximum temperature                                                                             |
| station_min_temp_c                  |  float    |  NOAA's GHCN      | Minimum temperature                                                                             |
| station_avg_temp_c                  |  float    |  NOAA's GHCN      | Average temperature                                                                             |
| station_precip_mm                   |  float    |  NOAA's GHCN      | Total precipitation                                                                             |
| station_diur_temp_rng_c             |  float    |  NOAA's GHCN      | Diurnal temperature range                                                                       |
| precipitation_amt_mm                |  float    |  PERSIANN satellite      | Total precipitation                                                                             |
| reanalysis_sat_precip_amt_mm        |  float    |  NOAA's NCEP      | Total precipitation                                                                             |
| reanalysis_dew_point_temp_k         |  float    |  NOAA's NCEP      | Mean dew point temperature                                                                      |
| reanalysis_air_temp_k               |  float    |  NOAA's NCEP      | Mean air temperature                                                                            |
| reanalysis_relative_humidity_percent|  float    |  NOAA's NCEP      | Mean relative humidity                                                                          |
| reanalysis_specific_humidity_g_per_kg |  float    |  NOAA's NCEP      | Mean specific humidity                                                                          |
| reanalysis_precip_amt_kg_per_m2     |  float    |  NOAA's NCEP      | Total precipitation                                                                             |
| reanalysis_max_air_temp_k           |  float    |  NOAA's NCEP      | Maximum air temperature                                                                         |
| reanalysis_min_air_temp_k           |  float    |  NOAA's NCEP      | Minimum air temperature                                                                         |
| reanalysis_avg_temp_k               |  float    |  NOAA's NCEP      | Average air temperature                                                                         |
| reanalysis_tdtr_k                   |  float    |  NOAA's NCEP      | Diurnal temperature range                                                                       |
| ndvi_se                             |  float    |  NOAA's CDR      | Pixel southeast of city centroid                                                                |
| ndvi_sw                             |  float    |  NOAA's CDR      | Pixel southwest of city centroid                                                                |
| ndvi_ne                             |  float    |  NOAA's CDR      | Pixel northeast of city centroid                                                                |
| ndvi_nw                             |  float    |  NOAA's CDR      | Pixel northwest of city centroid                                                                |

## Conclusion and Future Work
While precise prediction is difficult, the autoregressive XGBoost model appears to be able to capture major trends in dengue cases, as shown in the selected fold's performance in the training/validation plot below.

<img src=".\data\train_val_fold3.png" alt="Training and validation prediction vs ground truth comparison" width="800"/>

For future work, I would like to explore the following:
1. Auto-correlation in long-range forecasting - Time series split, as opposed to traditional K-fold cross validation approaches, is performed here to prevent data leakage. However, for the time scale in this scenario for long-range forecasting, the correlation between time points that are far enough apart should be minimal. The use of (unrandomized) K-fold cross validation could provide healthier train set sizes for a more effective parameter tuning process.
2. Traditional time series treatment - In this notebook, I have performed smoothing for the lagged feature values. In view of the proven track record of more traditional time-series processing (e.g. SARIMAX), I would like to apply moving averages and differencing to the lagged target values.
3. Ensemble approaches - Ensembling is an extremely powerful technique that could quickly improve model performance. The use multiple perturbed versions of the current XGBoost model, or of Catboost and LightGBM models, can be used to verify if this is the case for time series problems.
