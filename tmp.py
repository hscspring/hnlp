import json
import pnlp


lines = pnlp.read_lines("./tests/dataset/corpus_data.txt")

data = []

for line in lines:
    text, label = line.strip().split("|||")
    item = {}
    item["text"] = text
    item["label"] = label

    new = json.dumps(item, ensure_ascii=False)
    data.append(new)


pnlp.write_file("./tests/dataset/labeled_corpus.txt", data)
