from dask.distributed import Client, LocalCluster, as_completed
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def main():
    x = np.random.normal(size=(1000000, 5))
    y = x.mean(axis=1)
    cluster = LocalCluster(n_workers=4, threads_per_worker=1, memory_limit='1G')
    client = Client(cluster)
    print(client)
    print("scattering")
    [x_ref, y_ref] = client.scatter([x, y], broadcast=True)
    jobs = []
    for e in range(1, 30):
        print(e)
        jobs.append(client.submit(train_rf, e, x_ref, y_ref))
    for job in as_completed(jobs):
        print(job.result())
        del job
        client.rebalance()
    client.close()
    cluster.close()
    return


def train_rf(e, x, y):
    rf = RandomForestRegressor(n_estimators=e, max_depth=3, verbose=1)
    rf.fit(x,y)
    preds = rf.predict(x)
    print(preds)
    return preds.mean()

if __name__ == "__main__":
    main()
