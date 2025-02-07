import pandas as pd
from json import loads, load, dump

import matplotlib.pyplot as plt
from os import makedirs



class OtherCheck:

    def __init__(self, file_paths: list[str]):
        self.file_paths: list[str] = file_paths

    def comparison_full(self):

        data = {}

        for file_path in self.file_paths:
            with open(file_path, 'r') as f:
                data[file_path] = load(f)

        gpt_4o_mini = data[self.file_paths[0]]
        llama_3_1 = data[self.file_paths[1]]
        llama_3_7 = data[self.file_paths[2]]

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

        return df

    def save_df(self):
        json_str = self.comparison_full()
        json_str.to_csv('stats/output.csv', index=False)
        # json_data = loads(json_str)

        # with open('stats/statistics_simple.csv', 'w') as f:
        #     dump(json_str, f, indent=4)


class Analysis(OtherCheck):

    def __init__(self):
        super().__init__(file_paths=files)

    def analyze_and_visualize(self):
        df = pd.DataFrame(self.comparison_full())

        df['same_type'] = df.apply(
            lambda row: row['gpt_4o_mini_type'] == row['llama_3_1_type'] == row['llama_3_7_type'], axis=1)
        same_type_count = df['same_type'].sum()
        different_type_count = len(df) - same_type_count

        print(f"Number of notices with the same type: {same_type_count}")
        print(f"Number of notices with different types: {different_type_count}")

        df['same_status'] = df.apply(
            lambda row: row['gpt_4o_mini_status'] == row['llama_3_1_status'] == row['llama_3_7_status'], axis=1)
        same_status_count = df['same_status'].sum()
        different_status_count = len(df) - same_status_count

        print(f"Number of notices with the same status: {same_status_count}")
        print(f"Number of notices with different statuses: {different_status_count}")

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

        categories = ['Same Type', 'Different Type']
        counts = [same_type_count, different_type_count]
        axes[0].bar(categories, counts, color=['blue', 'orange'])
        axes[0].set_xlabel('Category')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Analysis of Notice Types')

        statuses = ['Same Status', 'Different Status']
        counts = [same_status_count, different_status_count]
        axes[1].bar(statuses, counts, color=['green', 'red'])
        axes[1].set_xlabel('Status')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Analysis of Notice Statuses')

        plt.tight_layout()
        makedirs('visualization', exist_ok=True)

        plt.savefig('visualization/analysis_visualization.png')
        plt.show()


if __name__ == '__main__':

    files = ['data/gpt_4o_mini.json', 'data/llama_3_1.json', 'data/llama_3_7.json']
    other_check = OtherCheck(file_paths=files)

    analysis = Analysis()
    analysis.analyze_and_visualize()


