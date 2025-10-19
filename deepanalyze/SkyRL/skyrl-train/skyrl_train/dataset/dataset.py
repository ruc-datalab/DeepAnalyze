import datasets
from loguru import logger
import os


class PromptDataset:
    def __init__(
        self,
        data_files,
        tokenizer: callable,
        max_prompt_length: int,
        num_workers: int = 8,
        prompt_key: str = "prompt",
        env_class_key: str = "env_class",
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.data_files = data_files
        self.prompt_key = prompt_key
        self.env_class_key = env_class_key
        self.num_workers = num_workers
        self._read_files_and_tokenize()

    def _read_files_and_tokenize(self):
        dataframes = []
        for data_file in self.data_files:
            ext = os.path.splitext(data_file)[-1].lower()
            if ext == ".parquet":
                dataset = datasets.load_dataset(
                    "parquet", data_files=data_file, keep_in_memory=True
                )["train"]
            elif ext == ".json":
                dataset = datasets.load_dataset(
                    "json", data_files=data_file, keep_in_memory=True
                )["train"]
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
            dataframes.append(dataset)

        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        logger.info(f"dataset len: {len(self.dataframe)}")

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key
        self.dataframe = self.dataframe.filter(
            lambda doc: len(
                tokenizer.apply_chat_template(
                    doc[prompt_key], add_generation_prompt=True
                )
            )
            <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
        )

        logger.info(f"filter dataset len: {len(self.dataframe)}")

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]
        messages = row_dict.pop(self.prompt_key)
        env_class = row_dict.pop(self.env_class_key, None)

        extra = {
            key: value
            for key, value in row_dict.items()
            if key != self.prompt_key and key != self.env_class_key
        }

        return messages, env_class, extra

    def collate_fn(self, item_list):
        all_inputs = []
        for prompt, env_class, env_extras in item_list:
            all_inputs.append(
                {"prompt": prompt, "env_class": env_class, "env_extras": env_extras}
            )
        return all_inputs

    def __len__(self):
        return len(self.dataframe)
