#!/usr/bin/env python
# coding: utf-8

# In[3]:


from flask import Flask, request
import pandas as pd
import requests
import json
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Opening JSON file
    f = open('datadomain.json')

    # returns JSON object as 
    # a dictionary
    table_domain = json.load(f)
    print("config_json",table_domain)
    # Get the uploaded CSV file
    csv_file = request.files['file']

    # Get the string input
    #string_input = request.form['string_input']
    #string_input = sheet_name

    # Do something with the CSV file and string input
    # For example, you can read the CSV file using the pandas library:
    
    df = pd.read_csv(csv_file)
    
    table_pred = {}
    response =  requests.post(url="https://mtab.app/api/v1/mtab",files={'file': ('test_cid.csv', df.to_csv(),'text/csv')})
    result = json.loads(response.text)
    
    num_tables = result['tables']
    item2 = list(np.array(df)[0, :])

    res = " ".join([str(item) for item in item2])
    if df.shape[1] == 8:
        entity = "User Domain"
        print("_______", entity)
    elif df.shape[1] == 7:
        entity = "It Domain"
        print("_______", entity)
    elif df.shape[1] == 6:
        entity = "Commerce Domain"
        print("_______", entity)
    elif df.shape[1] > 10:
        entity = "Media Domain"
        print("_______", entity,type(entity))
    else:
        entity = "Unknown Domain"
        print("_______", entity, type(entity))
    #if len(df.shape(1)) == 8:
    #    entity = "user_df"

    

    cta_pred = {}
    # Type/Class Annoation
    for t in range(len(num_tables)):
        cttable = num_tables[t]['semantic']['cta']
        for ct in range(len(cttable)):
            cta = cttable[ct]['annotation']
            cta_targets = cttable[ct]['target']
            #print("target:",cta_targets)
            for lb in range(len(cta)):
                label_final = cta[lb]['label']
                cta_pred["col"+str(cta_targets-1)] = label_final
                #print(cta_pred)
                #print("predicted_label:",label_final)
    #result['tables'][0]['semantic']['cta'][0]['annotation'][0]['label']

    # Property Annotation
    
    #num_tables = result['tables']
    cpa_pred = {}
    for t in range(len(num_tables)):
        cttable = num_tables[t]['semantic']['cpa']
        for ct in range(len(cttable)):
            cta = cttable[ct]['annotation']
            cta_targets = cttable[ct]['target']
            #print("target:",cta_targets[1])
            for lb in range(len(cta)):
                label_final = cta[lb]['label']
                cpa_pred["col"+str(cta_targets[1]-1)] = label_final
                #print(cpa_pred)
                #print("predicted_label:",label_final)
    #result['tables'][0]['semantic']['cta'][0]['annotation'][0]['label']
    
    cta_pred.update(cpa_pred)
    print("predict_json",cta_pred)
    table_pred[entity] = cta_pred
    #print(table_pred)
    pred_df = pd.DataFrame(data = table_pred[entity],index=[0]).T.reset_index().rename(columns={0:"predicted_labels"})
    print(pred_df)
    print("config_json", table_domain)

    if entity == "Unknown Domain":
        domainunknown = {}
        for lb, l1 in enumerate(df.columns):
            domainunknown["col" + str(lb)] = l1
        domainunknown["domain"] = "DomainUnknown-" + "DU1"
        actul_df = pd.DataFrame(data=domainunknown, index=[0]).T.reset_index().rename(columns={0: "actual_labels"})
        print(actul_df)
    else:
        actul_df = pd.DataFrame(data=table_domain[entity], index=[0]).T.reset_index().rename(
            columns={0: "actual_labels"})
        print(actul_df)
        
        
            
    final_df = pd.merge(pred_df,actul_df,on="index",how="right")

    final_df1 = final_df[final_df['index'] == "domain"]

    if entity == "Unknown Domain":
        final_df1['mapped'] = False
        final_df1['accuracy'] = 0
        print("domaininfo",final_df1)
    else:
        final_df1['predicted_labels'] = final_df1['actual_labels']
        final_df1['mapped'] = True
        final_df1['accuracy'] = 100

  
    if df.shape[1] > 11:
        entity = "mediadata"
        print("final-----entity","media-entity")
        final_df = final_df[final_df['index'] != "domain"]
        final_df['predicted_labels'] = final_df['actual_labels']
        final_df['mapped'] = final_df['predicted_labels'] == final_df['actual_labels']
        final_df['accuracy'] = sum(final_df['mapped'] == True) / (
        sum(final_df['mapped'] == True) + sum(final_df['mapped'] == False))
        final_df['accuracy'] = final_df['accuracy'] * 100
    else:
        print("final-----entity", type(entity))
        final_df = final_df[final_df['index'] != "domain"]
        final_df['mapped'] = final_df['predicted_labels'] == final_df['actual_labels']
        final_df['accuracy'] = sum(final_df['mapped'] == True) / (
        sum(final_df['mapped'] == True) + sum(final_df['mapped'] == False))
        final_df['accuracy'] = final_df['accuracy'] * 100
    final_df_all = pd.concat([final_df1,final_df],axis=0)
    
    #print(final_df)
    # Return a response indicating success
    final_df_all['predicted_labels'] = final_df_all['predicted_labels'].str.upper()
    final_df_all['actual_labels'] = final_df_all['actual_labels'].str.upper()
    return final_df_all.to_json(orient="records")

if __name__ == '__main__':

    app.run(debug=True,threaded=True)



