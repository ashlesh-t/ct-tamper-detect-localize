from utils.dataloader import DataLoader

class SharedDataLoader:
    def __init__(self, dataset_chunks, normdata_path, img_res):
        """
        dataset_chunks: list of .npy paths
        normdata_path: path to normalization files
        img_res: (rows, cols, depth)
        """
        self.dataset_chunks = dataset_chunks
        self.normdata_path = normdata_path
        self.img_res = img_res

    def load_batch(self, batch_size):
        """
        Generator over all batches from all chunks.
        Loads one chunk at a time to control memory.
        """
        for chunk_path in self.dataset_chunks:
            print(f"\n[SharedDataLoader] Loading chunk: {chunk_path}")
            loader = DataLoader(
                dataset_path=chunk_path,
                normdata_path=self.normdata_path,
                img_res=self.img_res
            )
            for batch in loader.load_batch(batch_size):
                yield batch
            del loader

    def load_data(self, batch_size, is_testing=False):
        """
        For show_progress / testing use
        Just loads from the *first* chunk
        """
        first_chunk = self.dataset_chunks[0]
        loader = DataLoader(
            dataset_path=first_chunk,
            normdata_path=self.normdata_path,
            img_res=self.img_res
        )
        return loader.load_data(batch_size, is_testing)
