import pandas as pd
from pafy import pafy


def createDictFromDF(df):
    res = dict()
    for url in df["URL"]:
        youtubeObj = pafy.new(url)
        ID = youtubeObj.videoid
        res[ID] = url
    return res


# function to compare LoadList files
def compare(file1, file2):
    loadList1 = pd.read_csv(file1)
    loadList2 = pd.read_csv(file2)
    d1 = createDictFromDF(loadList1)
    d2 = createDictFromDF(loadList2)
    id_set1 = set(d1.keys())
    id_set2 = set(d2.keys())
    intersection = id_set1.intersection(id_set2)
    int_url1 = set()
    int_url2 = set()
    for elem in intersection:
        int_url1.add(d1[elem])
        int_url2.add(d2[elem])
    loadList1["ORIGINAL"] = loadList1["URL"].apply(lambda x: x not in int_url1)
    loadList2["ORIGINAL"] = loadList2["URL"].apply(lambda x: x not in int_url2)

    loadList1.to_csv(file1, index=False)
    loadList2.to_csv(file2, index=False)


compare("list1.csv", "list2.csv")
