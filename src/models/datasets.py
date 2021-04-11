from typing import List, Optional

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


def read_usgs(
    filepath: str,
    columns: List[str] = ["datetime", "latitude", "longitude", "depth", "mag"],
    length_limit: Optional[int] = None,
) -> pd.DataFrame:

    assert filepath.endswith(".tsv")
    assert "datetime" in columns

    df = pd.read_csv(filepath, sep="\t", usecols=columns, nrows=length_limit)
    df["datetime"] = pd.to_datetime(df["datetime"])

    return df


def get_map_grids(n_rows, n_cols, coordinates):

    lats = np.linspace(90, -90, n_rows + 1)
    lons = np.linspace(-180, 180, n_cols + 1)

    coord_to_grid_indices = np.zeros((len(coordinates), 2))
    for i, coordinate in enumerate(coordinates):

        lat, lon = coordinate
        lat_idx = np.where(lat < lats)[0][-1]
        lon_idx = np.where(lon > lons)[0][-1]

        coord_to_grid_indices[i] = lat_idx, lon_idx
    return coord_to_grid_indices


class QuakeDataset(Dataset):
    def __init__(
        self,
        filepath,
        columns,
        grid_x_size,
        grid_y_size,
        num_points,
        biggest_q_in,  # biggest quake in `biggest_q_in` quakes
        length_limit=None,
    ):
        self.filepath = filepath
        self.columns = columns
        self.length_limit = length_limit

        self.grid_x_size = grid_x_size
        self.grid_y_size = grid_y_size
        self.num_points = num_points
        self.biggest_q_in = biggest_q_in

        assert "longitude" and "latitude" and "mag" in columns
        assert "datetime" in columns

        self.df = read_usgs(
            filepath=filepath, columns=columns, length_limit=length_limit
        )

        self.df.dropna(
            subset=["datetime", "latitude", "longitude", "mag"], inplace=True
        )

        self.df[["grid_x", "grid_y"]] = get_map_grids(
            n_rows=grid_x_size,
            n_cols=grid_y_size,
            coordinates=self.df[["latitude", "longitude"]].to_numpy(),
        )

        self.normalize_data()

    def normalize_data(
        self,
    ):
        self.df["latitude_norm"] = self.df["latitude"] / 90

        self.df["longitude_norm"] = self.df["longitude"] / 180

        self.df["mag_norm"] = (self.df["mag"] - 5) / 5

    def __len__(
        self,
    ):
        return len(self.df) - self.num_points - self.biggest_q_in + 1

    def __getitem__(self, idx):
        X = self.df[idx : idx + self.num_points][
            ["latitude_norm", "longitude_norm", "mag_norm"]
        ].to_numpy()

        y_data = self.df[
            idx + self.num_points : idx + self.num_points + self.biggest_q_in
        ][["grid_x", "grid_y", "mag"]]
        y_max_mag_idx = y_data["mag"].idxmax()

        y = y_data[["grid_x", "grid_y"]].loc[y_max_mag_idx]

        return torch.tensor(X).squeeze(), torch.tensor(y)
