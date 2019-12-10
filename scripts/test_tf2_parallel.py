import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential
from dask.distributed import Client, LocalCluster
from sklearn.metrics import mean_squared_error
import numpy as np

def main():
    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)
    print("started cluster")
    num_layers = [2, 3, 4]
    num_neurons = [20, 40, 60]
    futures = []
    for l in num_layers:
        for n in num_neurons:
            futures.append(client.submit(train_random_model, l, n))
    results = client.gather(futures)
    print(results)
    client.close()
    return


def train_random_model(num_layers, num_neurons):
    print(num_layers, num_neurons)
    n_inputs = 5
    examples = 10000
    x = np.random.normal(size=(examples, n_inputs)).astype(np.float32)
    y = x.prod(axis=1)
    mod = Sequential()
    mod.add(Input(shape=(n_inputs,)))
    for l in range(num_layers):
        mod.add(Dense(num_neurons))
    mod.add(Dense(1))
    mod.compile(optimizer="adam", loss="mse")
    mod.fit(x, y, epochs=30)
    preds = mod.predict(x)
    return mean_squared_error(y, preds)

if __name__ == "__main__":
    main()