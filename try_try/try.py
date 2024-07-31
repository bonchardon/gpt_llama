from json import loads

import pandas as pd


# todo: can be used to analize all the data given'
def comparison_full(self):
    with open('data/gpt_4o_mini.json', 'r') as f:
        gpt_4o_mini = loads(f.read())
        for i in gpt_4o_mini:
            i['verified'] = None

    with open('data/llama_3_1.json', 'r') as f:
        llama_3_1 = loads(f.read())
        for i in llama_3_1:
            i['verified'] = None

    with open('data/llama_3_7.json', 'r') as f:
        llama_3_7 = loads(f.read())
        for i in llama_3_7:
            i['verified'] = None

    df = pd.DataFrame({
        'gpt_4o_mini': gpt_4o_mini,
        'llama_3_1': llama_3_1,
        'llama_3_7': llama_3_7,
        'verified': None
    })

    for idx in range(len(df)):
        k = df.at[idx, 'gpt_4o_mini']
        l = df.at[idx, 'llama_3_1']
        i = df.at[idx, 'llama_3_7']

        if k['status'] == l['status'] == i['status'] and k['type'] == l['type'] == i['type']:
            df['verified'] = True
        else:
            df['verified'] = False

    return df