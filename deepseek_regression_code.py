import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Step 1: Data Preparation
# ------------------------------------------------
# Example datasets with 13 features
np.random.seed(42)
n_samples = 100
n_datasets = 3
n_features = 13

# Generate synthetic datasets
datasets = []
for dataset_id in range(1, n_datasets + 1):
    data = pd.DataFrame({
        **{f'feature{i}': np.random.randn(n_samples) for i in range(1, n_features + 1)},
        'target': np.random.randn(n_samples) * 10 + 50,  # Random target values
        'dataset_id': dataset_id
    })
    datasets.append(data)

# Combine datasets
data = pd.concat(datasets, ignore_index=True)

# Convert dataset_id to categorical
data['dataset_id'] = data['dataset_id'].astype('category')

# Step 2: Model Definition
# ------------------------------------------------
with pm.Model() as hierarchical_model:
    # Hyperpriors for global parameters
    mu_intercept = pm.Normal('mu_intercept', mu=0, sigma=10)  # Global intercept
    sigma_intercept = pm.HalfNormal('sigma_intercept', sigma=10)  # Variation in intercepts

    # Global slopes for each feature
    mu_slopes = pm.Normal('mu_slopes', mu=0, sigma=10, shape=n_features)  # Global slopes
    sigma_slopes = pm.HalfNormal('sigma_slopes', sigma=10, shape=n_features)  # Variation in slopes

    # Group-specific intercepts and slopes
    intercepts = pm.Normal('intercepts', mu=mu_intercept, sigma=sigma_intercept, shape=n_datasets)
    slopes = pm.Normal('slopes', mu=mu_slopes, sigma=sigma_slopes, shape=(n_datasets, n_features))

    # Model error
    sigma = pm.HalfNormal('sigma', sigma=10)

    # Linear model
    mu = intercepts[data['dataset_id'].cat.codes]
    for i in range(n_features):
        mu += slopes[data['dataset_id'].cat.codes, i] * data[f'feature{i + 1}']

    # Likelihood
    target = pm.Normal('target', mu=mu, sigma=sigma, observed=data['target'])

# Step 3: Model Training
# ------------------------------------------------
with hierarchical_model:
    # Sampling
    trace = pm.sample(2000, tune=1000, chains=2, return_inferencedata=True)

# Step 4: Model Evaluation
# ------------------------------------------------
# Summary of results
print(az.summary(trace))

# Diagnostics
az.plot_trace(trace)
plt.show()

# Forest plot for group-specific parameters
az.plot_forest(trace, var_names=['intercepts', 'slopes'])
plt.show()

# Step 5: Visualization
# ------------------------------------------------
# Plot posterior distributions of slopes and intercepts
az.plot_posterior(trace, var_names=['intercepts', 'slopes'])
plt.show()

# Step 6: Prediction
# ------------------------------------------------
# Generate new data for prediction
new_data = pd.DataFrame({
    **{f'feature{i}': np.random.randn(100) for i in range(1, n_features + 1)},
    'dataset_id': np.random.choice([1, 2, 3], 100)  # Randomly assign dataset_id
})

# Predict using the posterior samples
with hierarchical_model:
    pm.set_data({'dataset_id': new_data['dataset_id'].astype('category').cat.codes,
                 **{f'feature{i}': new_data[f'feature{i}'] for i in range(1, n_features + 1)}})
    posterior_predictive = pm.sample_posterior_predictive(trace, var_names=['target'])

# Add predictions to new_data
new_data['predicted_target'] = posterior_predictive['target'].mean(axis=0)

# Plot predictions
plt.figure(figsize=(10, 6))
for dataset_id in new_data['dataset_id'].unique():
    subset = new_data[new_data['dataset_id'] == dataset_id]
    plt.scatter(subset['feature1'], subset['predicted_target'], label=f'Dataset {dataset_id}')
plt.xlabel('Feature1')
plt.ylabel('Predicted Target')
plt.legend()
plt.title('Predictions for New Data')
plt.show()
