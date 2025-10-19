## Downloading SQL Data

We provide the training set we used on HuggingFace: https://huggingface.co/datasets/NovaSky-AI/SkyRL-SQL-653-data-newfmt 

You can download the dataset by running the following command:

```bash
huggingface-cli download NovaSky-AI/SkyRL-SQL-653-data-newfmt --local-dir $HOME/data/sql --repo-type dataset
```

## DB environment 

Next, download the data used to populate the database. We use the database files from [OmniSQL](https://github.com/RUCKBReasoning/OmniSQL/blob/main/train_and_evaluate/README.md):


- [ModelScope-OmniSQL-datasets](https://modelscope.cn/datasets/seeklhy/OmniSQL-datasets/summary)
- [HuggingFace-OmniSQL-datasets](https://huggingface.co/datasets/seeklhy/OmniSQL-datasets)


The dataset download is 22.2GB and include BIRD, Spider, ScienceBenchmark, EHRSQL, Spider2-SQLite, Spider-DK, Spider-Realistic, Spider-Syn, and SynSQL-2.5M. In our training pipeline, we only need to access databases from SynSQL-2.5M and Spider. 

To download, run:

```bash
huggingface-cli download seeklhy/OmniSQL-datasets data.zip --repo-type dataset --local-dir $HOME/data/sql/db_files/

unzip $HOME/data/sql/db_files/data.zip -d $HOME/data/sql/db_files/
```

If you modify the db_files path, update `DB_PATH` in `run_sql_fsdp.sh` accordingly.

## Training

See the bash scripts in this repo to launch training. Take a look at the file of interest, and modify any training configuration parameters as needed.

To start training, run:

```bash
bash examples/text_to_sql/<run_XXX>.sh
```