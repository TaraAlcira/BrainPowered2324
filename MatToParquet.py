import scipy.io as sio
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa


class MatToParquet:
    def __init__(self, mat_path, filename, folder):
        self.matpath = mat_path
        self.filename = filename
        self.folder = folder

        test = sio.loadmat(self.matpath)
        df = pd.DataFrame(test['data'])
        table = pa.Table.from_pandas(df)

        pq.write_table(table, f'{folder}/{filename}.parquet')
        print("done")

       