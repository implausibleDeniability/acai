class DataLoaderBase:
    def get_train_batch(self):
        raise NotImplementedError()

    def get_eval_batch(self):
        raise NotImplementedError()
