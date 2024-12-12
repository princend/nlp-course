from datasets import load_dataset

squad_it_dataset = load_dataset("json", data_files="SQuAD_it-train.json", field="data")

# print(squad_it_dataset)
print(squad_it_dataset["train"][0])


### 映射

## 方案一使用json
# data_files = {"train": "SQuAD_it-train.json", "test": "SQuAD_it-test.json"}
# squad_it_dataset = load_dataset("json", data_files=data_files, field="data")

## 方案二使用json.gz
# data_files = {"train": "SQuAD_it-train.json.gz", "test": "SQuAD_it-test.json.gz"}
# squad_it_dataset = load_dataset("json", data_files=data_files, field="data")

## 方案三使用url
url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")

print(squad_it_dataset)