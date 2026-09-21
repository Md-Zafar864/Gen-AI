age=[23,34,22,35]

work=[2,77,7]

record=[]

for i, j in enumerate(age):
    record.append({
        "id":i,
        "value":j,
        "age":work[i]
    })

print(record)