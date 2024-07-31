import pandas as pd

from json import loads, load, dumps, dump


class OtherCheck:

    def open_files(self, file):
        with open(file, 'r') as w:
            data = load(w)

        documents = data.get("documents", [])
        with open('data/gpt_4o_mini.json', 'w') as output_file:
            dump(documents, output_file, indent=4)

    def comparison(self):
        file1 = pd.read_json('data/gpt_4o_mini.json')
        file2 = pd.read_json('data/llama_3_1.json')
        file3 = pd.read_json('data/llama_3_7.json')

        df_1 = pd.DataFrame(file1)
        df_1['source'] = 'gpt_4o_mini'
        df_2 = pd.DataFrame(file2)
        df_2['source'] = 'llama_3_1'
        df_3 = pd.DataFrame(file3)
        df_3['source'] = 'llama_3_7'

        concat_df = pd.concat([df_1, df_2, df_3], ignore_index=True)
        print(concat_df)
        answer_df = concat_df.drop_duplicates(subset=['status'], keep=False)
        return answer_df

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


    def comparison_full_2(self):

        with open('data/gpt_4o_mini.json', 'r') as f:
            gpt_4o_mini = loads(f.read())

        with open('data/llama_3_1.json', 'r') as f:
            llama_3_1 = loads(f.read())

        with open('data/llama_3_7.json', 'r') as f:
            llama_3_7 = loads(f.read())

        df = pd.DataFrame({
            'id': [i['id'] for i in gpt_4o_mini],
            'gpt_4o_mini_status': [i['status'] for i in gpt_4o_mini],
            'llama_3_1_status': [i['status'] for i in llama_3_1],
            'llama_3_7_status': [i['status'] for i in llama_3_7],
            'gpt_4o_mini_type': [i['type'] for i in gpt_4o_mini],
            'llama_3_1_type': [i['type'] for i in llama_3_1],
            'llama_3_7_type': [i['type'] for i in llama_3_7],
            'verified': None
        })

        status_check = (df['gpt_4o_mini_status'] == df['llama_3_1_status']) & (
                    df['gpt_4o_mini_status'] == df['llama_3_7_status'])
        type_check = (df['gpt_4o_mini_type'] == df['llama_3_1_type']) & (df['gpt_4o_mini_type'] == df['llama_3_7_type'])

        verified_check = status_check & type_check
        df['verified'] = verified_check

        print("Rows where 'verified' is False:")
        print(df[~df['verified']])

        json_str = df.to_json(orient='records')
        json_data = loads(json_str)

        with open('stats/statistics.csv', 'w') as f:
            dump(json_data, f, indent=4)

        return df


if __name__ == '__main__':
    # data = open_files('data/gpt_4o_mini.json')
    # print(data)
    other_check = OtherCheck()
    print(other_check.comparison_full_2())

