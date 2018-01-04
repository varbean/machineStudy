import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False
#读取索引
def beginRead():
    emails=[]

    with open("data/full/index","r",encoding="gbk",errors="ignore") as r:
        ris=r.readlines()
        for ri in ris:
            email = {}
            ri=ri.replace("\n","").split("../")
            if ri[0].find("spam") ==-1:
                email["sflj"]="1"
            else:
                email["sflj"]="0"
            readEmail(email,ri[1]) #读取具体邮件的文件。
            result  =email.get("From","unknown").replace(",","").strip()+","
            result += email.get("To", "unknown").replace(",", "").strip() + ","
            result += email.get("Date", "unknown").replace(",", "").strip() + ","
            result += email.get("content", "unknown").replace(",", "").strip() + ","+email.get("sflj", "unknown")
            emails.append(result)
    return emails

#读取邮件内容
def readEmail(email,path):
    with open(path,"r",encoding="gbk",errors="ignore") as res:
        flag = False
        for re in res:
            re=re.strip()
            if re.startswith("From"):
                email["From"]=re[5:]
            elif re.startswith("To"):
                email["To"] = re[3:]
            elif re.startswith("Date"):
                email["Date"] = re[5:]
            elif not re:
                flag=True

            if flag:
                email["content"]=email["content"]+re.strip() if "content" in email else re.strip()

#将处理好的数据写入文件，方便后续处理
def write_result():
    emails=beginRead()
    with open("data/result","w+",encoding="utf-8") as writer:
        for email in emails:
            writer.writelines(email+"\n")






"""
    特征提取：先是人为辨别，然后再机器提取
    略
"""


def precess_content_length(lg):
    if lg <= 10:
        return 0
    elif lg <= 100:
        return 1
    elif lg <= 500:
        return 2
    elif lg <= 1000:
        return 3
    elif lg <= 1500:
        return 4
    elif lg <= 2000:
        return 5
    elif lg <= 2500:
        return 6
    elif lg <=  3000:
        return 7
    elif lg <= 4000:
        return 8
    elif lg <= 5000:
        return 9
    elif lg <= 10000:
        return 10
    elif lg <= 20000:
        return 11
    elif lg <= 30000:
        return 12
    elif lg <= 50000:
        return 13
    else:
        return 14
def test_feature():
    df=pd.read_csv("data/result",sep=",",encoding="utf-8",header=None,names=["from","to","date","content","label"])

    df['content_length'] = pd.Series(map(lambda st: len(str(st)), df['content']))
    df['content_length_type'] = pd.Series(map(lambda st: precess_content_length(st), df['content_length']))
    print("==========df========")
    print(df.head(5))
    df2 = df.groupby(['content_length_type', 'label'])['label'].agg(['count']).reset_index()
    print("==========df2========")
    print(df2.head(5))
    df3 = df2[df2.label == 1][['content_length_type', 'count']].rename(columns={'count':'c1'})
    print("==========df3========")
    print(df3.head(5))
    df4 = df2[df2.label == 0][['content_length_type', 'count']].rename(columns={'count':'c2'})
    print("==========df4========")
    print(df4.head(5))
    df5 = pd.merge(df3, df4)
    df5['c1_rage'] = df5.apply(lambda r: r['c1'] / (r['c1'] + r['c2']), axis=1)
    df5['c2_rage'] = df5.apply(lambda r: r['c2'] / (r['c1'] + r['c2']), axis=1)
    print("==========df5========")
    print(df5.head(20))
    # 画图
    plt.plot(df5['content_length_type'], df5['c1_rage'], label=u'垃圾邮件比例')
    plt.plot(df5['content_length_type'], df5['c2_rage'], label=u'正常邮件比例')
    plt.grid(True)
    plt.legend(loc = 0)
    plt.show()

# df=pd.read_csv("data/result",sep=",",encoding="utf-8",header=None,names=["from","to","date","content","label"])
# for i,d in enumerate(df["content"]):
#     if(type(d)!=str):
#         print(i)
test_feature()
