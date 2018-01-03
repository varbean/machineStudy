import numpy as np
import pandas as pd
import time

#读取索引
def beginRead():
    emails=[]

    ris=open("data/full/index","r",encoding="gbk",errors="ignore").readlines()
    for ri in ris:
        email = {}
        ri=ri.replace("\n","").split("../")
        if ri[0].find("spam") ==-1:
            email["sflj"]="1"
        else:
            email["sflj"]="0"
        print(ri[1])
        emails.append(email)

#读取邮件内容
def readEmail(email,path):
    res=open(path,"r",encoding="gbk",errors="ignore")
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
    print(email)
    res.close()

email={}
readEmail(email,"data/000/000")
