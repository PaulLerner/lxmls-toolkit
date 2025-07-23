import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


N = 1000
D_true = 3
D_obs = 200
N_train = 100
N_test = N - N_train

alphas = [1e-18, 1e-9, 1e-6, 1e-4, 1e-2, 1, 1e2]
ks = [1, 2, 5, 10, 20, 30, 50, 70, 80, 90, 100, 110, 120, 150, 170, 200]
n_samples = 10

train_rmse_matrix = np.zeros((len(alphas), len(ks)))
test_rmse_matrix = np.zeros((len(alphas), len(ks)))

def generate_data():
    X_true = np.random.randn(N, D_true)
    w_true = np.random.randn(D_true)
    y = X_true @ w_true + np.random.randn(N) * 0.1
    T = np.random.randn(D_true, D_obs)
    X_obs = X_true @ T + np.random.randn(N, D_obs) * 0.5

    return X_obs, y

for i, alpha in enumerate(alphas):
    print(f'alpha: {alpha}')
    for j, k in enumerate(ks):
        train_rmses = []
        test_rmses = []
        for sample_id in range(n_samples):
            X_obs, y = generate_data()
            X_train, X_test, y_train, y_test = train_test_split(
                X_obs, y, train_size=N_train, test_size=N_test, random_state=sample_id)

            X_train_k = X_train[:, :k]
            X_test_k = X_test[:, :k]

            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(X_train_k, y_train)

            y_train_pred = model.predict(X_train_k)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            train_rmses.append(train_rmse)

            y_test_pred = model.predict(X_test_k)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            test_rmses.append(test_rmse)

        train_rmse_matrix[i, j] = np.mean(train_rmses)
        test_rmse_matrix[i, j] = np.mean(test_rmses)

np.savetxt('train.txt', train_rmse_matrix)
np.savetxt('test.txt', test_rmse_matrix)

print("Saved train.txt and test.txt")


plt.figure(figsize=(12, 8))

for i, alpha in enumerate(alphas):
    plt.plot(ks, test_rmse_matrix[i, :], marker='.', linestyle='-', label=f'alpha={alpha:.0e}')

plt.axvline(x=N_train, linestyle='--', color='r', label=f'Interpolation threshold (k=N_train={N_train})')
plt.title('Double Descent: Test RMSE vs. Number of Features for Ridge Regression')
plt.xlabel('Number of features (k)')
plt.ylabel('Test RMSE (log scale)')
plt.legend(title='Regularization (alpha)')
plt.grid(True, which="both", ls="--")
plt.yscale('log')
plt.tight_layout()
plt.show()
