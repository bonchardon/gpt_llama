import pandas as pd

from json import loads, load, dumps, dump


class OtherCheck:

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

