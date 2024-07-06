from torch.utils.data import Dataset

from .data_sample import DataSample


class DatasetWrapper(Dataset):
    def __init__(self, wrapped_data: Dataset) -> None:
        self.wrapped_data = wrapped_data

    def __len__(self) -> int:
        return len(self.wrapped_data)  # type: ignore

    def __getitem__(self, idx: int) -> DataSample:
        batch = self.wrapped_data[idx]
        return DataSample(
            input=batch[0],
            target=batch[1],
        )
