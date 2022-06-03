with open("rawResultsCudaShared.txt") as f:
    results = f.readlines()

print(results)

resDict = dict()

for i, string in enumerate(results):
    if string.endswith(".pgm\n"):
        valueHost = float(results[i+1].split("Host processing time: ")[1].split(" (ms)")[0])
        valueCudaShared = float(results[i+2].split("Device processing time: ")[1].split(" (ms)")[0])
        efficiency = (valueHost - valueCudaShared) / valueHost * 100
        resDict[string[:-1]] = {"valueHost":valueHost, "valueCudaShared":valueCudaShared, "efficiency" : f"{efficiency:.{2}f}\%"}

print(resDict)

with open("statsTableTemplate.sty") as f:
    table_str = "".join(f.readlines()) + "\n"

for k, v in resDict.items():
    table_str+= "\multicolumn{{1}}{{|l|}}{{ {} }} & \multicolumn{{1}}{{l|}}{{ {} }} & \multicolumn{{1}}{{l|}}{{ {} }} & \multicolumn{{1}}{{l|}}{{ {} }} &  \\\ \cline{{1-4}} \n".format(k, v["valueHost"], v["valueCudaShared"], v["efficiency"]) # usou-se uma "\" a mais e "\n" no fim

table_str+= "\n \end{tabular} \n \end{table} \n"

with open("statsTableResults.sty","w") as f:
    f.write(table_str)
