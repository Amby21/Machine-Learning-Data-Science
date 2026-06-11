#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
#%%Data import

df = pd.read_csv("application_train.csv")
df.shape
columns = df.columns.to_list()
#%% target class distribution%%
df['TARGET'].value_counts()
#%% null values%%
df.isnull().sum().sort_values(ascending=False)
# %%
df_cols = pd.read_csv("HomeCredit_columns_description.csv", encoding='cp1252')

def get_col_desc(col_name: str) -> str:
    match = df_cols[df_cols['Row'] == col_name]['Description']
    return match.values[0] if not match.empty else ""
            
# %%
print(get_col_desc("EXT_SOURCE_3"))
# %%
df['COMMONAREA_MISSING'] = df['COMMONAREA_AVG'].isnull().astype(int)
#%% Missing value analysis - all columns
missing = df.isnull().sum()
#%%
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'missing_count': missing,
    'missing_pct': missing_pct
}).sort_values('missing_pct', ascending=False)

print(missing_df[missing_df['missing_pct'] > 0].head(20))
# %% dropping cols with suffix _MEDI/ MODE
cols_to_drop = [col for col in df.columns if col.endswith('_MODE') or col.endswith('_MEDI')]
df = df.drop(cols_to_drop, axis=1)

print(f"Columns dropped: {len(cols_to_drop)}")
print(f"Remaining columns: {df.shape[1]}")
# %%
# Group 1 - No car: Missing is expected, impute with 0
# Group 2 - Has car but missing age: This is MNAR - suspicious!

#%%
# DAYS_EMPLOYED anomaly — retired people flagged as 365243
df['DAYS_EMPLOYED_ANOMALY'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
#%%
# EXT_SOURCE — high missing, high signal → flag + impute
df['EXT_SOURCE_1_MISSING'] = df['EXT_SOURCE_1'].isnull().astype(int)
df['EXT_SOURCE_3_MISSING'] = df['EXT_SOURCE_3'].isnull().astype(int)
df['EXT_SOURCE_1'] = df['EXT_SOURCE_1'].fillna(df['EXT_SOURCE_1'].median())
df['EXT_SOURCE_3'] = df['EXT_SOURCE_3'].fillna(df['EXT_SOURCE_3'].median())

#%%
df.loc[df['FLAG_OWN_CAR'] == 'N', 'OWN_CAR_AGE'] = 0
median_car_age = df[df['FLAG_OWN_CAR'] == 'Y']['OWN_CAR_AGE'].median()
df.loc[(df['FLAG_OWN_CAR'] == 'Y') & 
       (df['OWN_CAR_AGE'].isnull()), 'OWN_CAR_AGE'] = median_car_age

#%%
enquiry_cols = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
                'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
                'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']
df[enquiry_cols] = df[enquiry_cols].fillna(0)

# DAYS_EMPLOYED — impute after anomaly replaced with NaN
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].fillna(df['DAYS_EMPLOYED'].median())
#%%
# Remaining numerical → median
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Remaining categorical → mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

print(f"Missing values remaining: {df.isnull().sum().sum()}")
# %% FEATURE ENGINEERING
df['AGE_YEARS']      = df['DAYS_BIRTH'] / -365
df['YEARS_EMPLOYED'] = df['DAYS_EMPLOYED'] / -365

#%%
# Credit burden ratios
df['DEBT_TO_INCOME']    = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
df['ANNUITY_TO_INCOME'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
df['CREDIT_TO_ANNUITY'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
# Stability ratio
df['EMPLOYMENT_RATIO'] = df['YEARS_EMPLOYED'] / df['AGE_YEARS']
# Credit enquiry aggregate
df['TOTAL_CREDIT_ENQUIRIES'] = df[enquiry_cols].sum(axis=1)
# EXT_SOURCE aggregates
df['EXT_SOURCE_MEAN']     = df[['EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
df['EXT_SOURCE_MIN']      = df[['EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
df['EXT_SOURCE_WEIGHTED'] = (0.5 * df['EXT_SOURCE_2'] +
                              0.3 * df['EXT_SOURCE_3'] +
                              0.2 * df['EXT_SOURCE_1'])

# %%BEHAVIORAL FEATURES
# ============================================================

df['APPLIED_ON_WEEKEND']  = df['WEEKDAY_APPR_PROCESS_START'].isin( ['SATURDAY', 'SUNDAY']).astype(int)
df['APPLIED_START_WEEK']  = df['WEEKDAY_APPR_PROCESS_START'].isin(['MONDAY', 'TUESDAY']).astype(int)

# %%ENCODING
# ============================================================

# Fix CODE_GENDER anomaly
df['CODE_GENDER'] = df['CODE_GENDER'].replace('XNA', df['CODE_GENDER'].mode()[0])

# Label encode all categoricals
le = LabelEncoder()
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

print(f"Final shape: {df.shape}")
print(f"Remaining object columns: {df.select_dtypes(include=['object']).shape[1]}")


# %%
from sklearn.model_selection import train_test_split
X = df.drop(['TARGET','SK_ID_CURR'],axis = 1)
y = df['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# Verify stratification worked
print(f"Train default rate: {y_train.mean():.2%}")
print(f"Test default rate:  {y_test.mean():.2%}")
print(f"\nTrain shape: {X_train.shape}")
print(f"Test shape:  {X_test.shape}")


# %%
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=0.5, random_state = 42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"Before SMOTE - Train shape: {X_train.shape}")
print(f"After SMOTE  - Train shape: {X_train_smote.shape}")
print(f"\nBefore SMOTE - Default rate: {y_train.mean():.2%}")
print(f"After SMOTE  - Default rate: {y_train_smote.mean():.2%}")
print(f"\nClass distribution after SMOTE:")
print(pd.Series(y_train_smote).value_counts())
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')
#%% Logistic Regression - Linear model
lr = LogisticRegression(class_weight='balanced',max_iter = 1000, random_state=42)
lr.fit(X_train_smote,y_train_smote)
lr_pred_proba = lr.predict_proba(X_test)[:,1]
lr_auc = roc_auc_score(y_test,lr_pred_proba)
print(f"Logistic Regression AUC: {lr_auc:.4f}")

#%% Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=8,class_weight=None,random_state=42, n_jobs=-1)
rf.fit(X_train_smote,y_train_smote)
rf_pred_proba = rf.predict_proba(X_test)[:,1]
rf_auc = roc_auc_score(y_test,rf_pred_proba)
print(f"Random Forest AUC:       {rf_auc:.4f}")
#%% XG Boost
xgb = XGBClassifier(n_estimators = 200, max_depth=6, learning_rate=0.05,scale_pos_weight=1, random_state = 42, eval_metric = 'auc',verbosity=0 )
xgb.fit(X_train_smote,y_train_smote)
xgb_pred_proba = xgb.predict_proba(X_test)[:,1]
xgb_auc = roc_auc_score(y_test, xgb_pred_proba)
print(f"XGBoost AUC:             {xgb_auc:.4f}")

# %%
from sklearn.metrics import (roc_auc_score, classification_report, 
                             confusion_matrix, RocCurveDisplay,
                             PrecisionRecallDisplay)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

#%%
precision, recall, thresholds = precision_recall_curve(y_test, xgb_pred_proba)

f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"Best F1 score:     {f1_scores[optimal_idx]:.4f}")

#%%
xgb_pred_optimal = (xgb_pred_proba >= optimal_threshold).astype(int)

print("\n=== XGBoost Classification Report (Optimal Threshold) ===")
print(classification_report(y_test, xgb_pred_optimal, 
      target_names=['No Default', 'Default']))
#%%

cm = confusion_matrix(y_test, xgb_pred_optimal)
print("\nConfusion Matrix:")
print(f"                 Predicted No Default  Predicted Default")
print(f"Actual No Default      {cm[0][0]:6d}               {cm[0][1]:6d}")
print(f"Actual Default         {cm[1][0]:6d}               {cm[1][1]:6d}")

#%%
tn, fp, fn, tp = cm.ravel()
print(f"\n=== Business Impact ===")
print(f"True Negatives  (correctly approved): {tn:6d}")
print(f"False Positives (good customer rejected): {fp:6d}")
print(f"False Negatives (defaulter approved): {fn:6d}")
print(f"True Positives  (defaulter caught):   {tp:6d}")
print(f"\nDefaulters caught:     {tp/(tp+fn):.2%} (Recall)")
print(f"Precision on defaults: {tp/(tp+fp):.2%}")

# %%
conservative_threshold = 0.2
xgb_pred_conservative = (xgb_pred_proba >= conservative_threshold).astype(int)

# from sklearn.metrics import confusion_matrix, classification_report
cm2 = confusion_matrix(y_test, xgb_pred_conservative)
tn, fp, fn, tp = cm2.ravel()

print(f"=== Conservative Bank Threshold (0.2) ===")
print(f"Defaulters caught:        {tp/(tp+fn):.2%} (Recall)")
print(f"Precision on defaults:    {tp/(tp+fp):.2%}")
print(f"Good customers rejected:  {fp}")
print(f"Defaulters approved:      {fn}")

# Also try lenient threshold
lenient_threshold = 0.5
xgb_pred_lenient = (xgb_pred_proba >= lenient_threshold).astype(int)
cm3 = confusion_matrix(y_test, xgb_pred_lenient)
tn2, fp2, fn2, tp2 = cm3.ravel()

print(f"\n=== Aggressive Lender Threshold (0.5) ===")
print(f"Defaulters caught:        {tp2/(tp2+fn2):.2%} (Recall)")
print(f"Precision on defaults:    {tp2/(tp2+fp2):.2%}")
print(f"Good customers rejected:  {fp2}")
print(f"Defaulters approved:      {fn2}")

# %%
# Check what predict_proba scores actually look like
print("Distribution of predicted probabilities:")
print(pd.Series(xgb_pred_proba).describe())
print(f"\nScores below 0.2:  {(xgb_pred_proba < 0.2).sum()}")
print(f"Scores 0.2 - 0.28: {((xgb_pred_proba >= 0.2) & (xgb_pred_proba < 0.28)).sum()}")
print(f"Scores 0.28 - 0.5: {((xgb_pred_proba >= 0.28) & (xgb_pred_proba < 0.5)).sum()}")
print(f"Scores above 0.5:  {(xgb_pred_proba > 0.5).sum()}")
# %%
# Visualize the full threshold curve
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

precision, recall, thresholds = precision_recall_curve(y_test, xgb_pred_proba)

# Find optimal F1 threshold
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"At optimal threshold:")
print(f"  Precision: {precision[optimal_idx]:.4f}")
print(f"  Recall:    {recall[optimal_idx]:.4f}")
print(f"  F1:        {f1_scores[optimal_idx]:.4f}")

# Plot Precision-Recall tradeoff
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(thresholds, precision[:-1], 'b-', label='Precision')
plt.plot(thresholds, recall[:-1], 'r-', label='Recall')
plt.plot(thresholds, f1_scores[:-1], 'g-', label='F1')
plt.axvline(x=optimal_threshold, color='black', linestyle='--', 
            label=f'Optimal={optimal_threshold:.2f}')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall-F1 vs Threshold')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(recall[:-1], precision[:-1], 'b-')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)

plt.tight_layout()
plt.savefig('threshold_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nThreshold Analysis:")
print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Flagged':>10}")
for thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    preds = (xgb_pred_proba >= thresh).astype(int)
    from sklearn.metrics import precision_score, recall_score, f1_score
    if preds.sum() > 0:
        p = precision_score(y_test, preds)
        r = recall_score(y_test, preds)
        f = f1_score(y_test, preds)
        flagged = preds.sum()
        print(f"{thresh:>10.2f} {p:>10.4f} {r:>10.4f} {f:>10.4f} {flagged:>10}")
# %% LightGBM

from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(
    n_estimators = 500,
    learning_rate = 0.05,
    max_depth = 6,
    num_leaves = 31,
    min_child_samples = 20,
    class_weight = None,
    random_state = 42,
    n_jobs = -1,
    verbose = -1
)

lgbm.fit(X_train_smote, y_train_smote)
lgbm_pred_proba = lgbm.predict_proba(X_test)[:,1]
lgbm_auc = roc_auc_score(y_test,lgbm_pred_proba)
print(f"Logistic Regression AUC: {lr_auc:.4f}")
print(f"Random Forest AUC:       {rf_auc:.4f}")
print(f"XGBoost AUC:             {xgb_auc:.4f}")
print(f"LightGBM AUC:            {lgbm_auc:.4f}")
# %% Hyper parameter tuning

from sklearn.model_selection import RandomizedSearchCV
import numpy as np

param_dist = {
    'n_estimators':    [200, 300, 500, 700, 1000],
    'learning_rate':   [0.01, 0.03, 0.05, 0.1, 0.2],
    'max_depth':       [4, 5, 6, 7, 8],
    'num_leaves':      [20, 31, 50, 70, 100],
    'min_child_samples': [10, 20, 30, 50],
    'subsample':       [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree':[0.6, 0.7, 0.8, 0.9, 1.0]
}

lgbm_base = LGBMClassifier( random_state = 42, n_jobs = -1, verbose= -1)

random_search = RandomizedSearchCV(
    estimator=lgbm_base,
    param_distributions=param_dist,
    n_iter=50,
    scoring = 'roc_auc',
    cv = 3,
    random_state=42,
    n_jobs = -1,
    verbose=1
)
print("Starting RandomizedSearch... (3-5 minutes)")
random_search.fit(X_train_smote, y_train_smote)
print(f"\nBest AUC (CV): {random_search.best_score_:.4f}")
print(f"\nBest parameters:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")
# %%
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV

# Pipeline ensures SMOTE happens INSIDE each fold
pipeline = Pipeline([
    ('smote', SMOTE(sampling_strategy=0.5, 
                   random_state=42)),
    ('model', LGBMClassifier(random_state=42,
                             n_jobs=-1,
                             verbose=-1))
])

# Parameter names must include step name
param_dist_pipe = {
    'model__n_estimators':      [200, 300, 500, 700, 1000],
    'model__learning_rate':     [0.01, 0.03, 0.05, 0.1, 0.2],
    'model__max_depth':         [4, 5, 6, 7, 8],
    'model__num_leaves':        [20, 31, 50, 70, 100],
    'model__min_child_samples': [10, 20, 30, 50],
    'model__subsample':         [0.6, 0.7, 0.8, 0.9, 1.0],
    'model__colsample_bytree':  [0.6, 0.7, 0.8, 0.9, 1.0]
}

random_search_pipe = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist_pipe,
    n_iter=50,
    scoring='roc_auc',
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("Starting correct pipeline search... (5-8 minutes)")
random_search_pipe.fit(X_train, y_train)  # ← raw X_train, not SMOTE version!

print(f"\nBest AUC (CV, no leakage): {random_search_pipe.best_score_:.4f}")
print(f"\nBest parameters:")
for param, value in random_search_pipe.best_params_.items():
    print(f"  {param}: {value}")
# %%
best_model = random_search_pipe.best_estimator_

# predict on real data
tuned_pred_proba = best_model.predict_proba(X_test)[:,1]
tuned_auc = roc_auc_score(y_test, tuned_pred_proba)

print("=== Final Model Comparison ===")
print(f"Logistic Regression:  {lr_auc:.4f}")
print(f"Random Forest:        {rf_auc:.4f}")
print(f"XGBoost:              {xgb_auc:.4f}")
print(f"LightGBM (baseline):  {lgbm_auc:.4f}")
print(f"LightGBM (tuned):     {tuned_auc:.4f}")
# %%
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test,tuned_pred_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"\nOptimal threshold: {optimal_threshold:.4f}")
print(f"At optimal threshold:")
print(f"  Precision: {precision[optimal_idx]:.4f}")
print(f"  Recall:    {recall[optimal_idx]:.4f}")
print(f"  F1:        {f1_scores[optimal_idx]:.4f}")
# %% SHAP
import shap

#lightgbm built-in SHAP
explainer = shap.TreeExplainer(best_model.named_steps['model'])

X_test_sample  = X_test.sample(1000, random_state = 42)
shap_values = explainer.shap_values(X_test_sample)

shap_values_default = shap_values

print(f"SHAP values shape: {shap_values_default.shape}")
print(f"X_test_sample shape: {X_test_sample.shape}")

#%%PLOT1: which feature matters most

plt.figure(figsize=(10,8))
shap.summary_plot(shap_values_default,X_test_sample,plot_type='bar',max_display=20, show=False)
plt.title('Global Feature Importance (SHAP)')
plt.tight_layout()
plt.savefig('shap_importance.png', dpi=150, bbox_inches='tight')
plt.show()
# %%
# Check what shap_values actually contains
print(f"Type of shap_values: {type(shap_values)}")

if isinstance(shap_values, list):
    print(f"Number of classes: {len(shap_values)}")
    print(f"Shape of class 0: {shap_values[0].shape}")
    print(f"Shape of class 1: {shap_values[1].shape}")
else:
    print(f"Shape of shap_values: {shap_values.shape}")
# %%
#find high risk customer
high_risk_idx = pd.Series(best_model.predict_proba(X_test_sample)[:,1]).nlargest(1).index[0]

print(f"Explaining customer at index: {high_risk_idx}")
print(f"Default probability: {best_model.predict_proba(X_test_sample)[:, 1][high_risk_idx]:.4f}")

# Waterfall plot — shows exactly why this person was flagged
shap.plots._waterfall.waterfall_legacy(
    explainer.expected_value,
    shap_values_default[high_risk_idx],
    X_test_sample.iloc[high_risk_idx],
    max_display=15,
    show=False
)
plt.title('Why Was This Customer Flagged as High Risk?')
plt.tight_layout()
plt.savefig('shap_local.png', dpi=150, bbox_inches='tight')
plt.show()
# %%
# BUSINESS EXPLANATION — Convert to human readable
# ============================================================
feature_shap = pd.DataFrame({
    'feature': X_test_sample.columns,
    'shap_value': shap_values_default[high_risk_idx],
    'feature_value': X_test_sample.iloc[high_risk_idx].values
}).sort_values('shap_value', ascending=False)

print("\n=== Top Risk Factors for This Customer ===")
print(feature_shap[feature_shap['shap_value'] > 0].head(5).to_string(index=False))
print("\n=== Top Protective Factors for This Customer ===")
print(feature_shap[feature_shap['shap_value'] < 0].head(5).to_string(index=False))
# %%
