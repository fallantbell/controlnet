# from inf_zero_dataset import MyDataset
from acid_dataset import MyDataset

dataset = MyDataset(mode="validation")
print(len(dataset))

item = dataset[130]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
begin = item['begin']
end = item['end']
range = item['range']
print(txt)
print(jpg.shape)
print(hint.shape)
print(begin.shape)
print(end.shape)
print(range)
