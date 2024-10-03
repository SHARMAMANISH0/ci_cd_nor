from sklearn.preprocessing import MinMaxScaler
import numpy as np

def normalize_data(data):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data

if __name__ == "__main__":
    data = np.array([[10, 200], [5, 100], [8, 150]])
    print("Original Data:", data)
    print("Normalized Data:", normalize_data(data))
