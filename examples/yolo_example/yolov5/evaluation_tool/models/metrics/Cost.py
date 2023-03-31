import pandas as pd


class Cost:
    url = 'https://raw.githubusercontent.com/full-stack-deep-learning/website/main/docs/cloud-gpus/cloud-gpus.csv'
    gcp = pd.read_csv(url, index_col=0, parse_dates=[0]).query('Cloud.str.startswith("GCP")',
                                                               engine="python").sort_values('On-demand',
                                                                                            ascending=True)
    aws = pd.read_csv(url, index_col=0, parse_dates=[0]).query('Cloud.str.startswith("AWS")',
                                                               engine="python").sort_values('On-demand',
                                                                                            ascending=True)
    azure = pd.read_csv(url, index_col=0, parse_dates=[0]).query('Cloud.str.startswith("Azure")',
                                                                 engine="python").sort_values('On-demand',
                                                                                              ascending=True)
    gpu = None

    def find_gpu(self, memory, cloud, type):
        if cloud == 'GCP':
            df = self.gcp
        elif cloud == 'AWS':
            df = self.aws
        elif cloud == "Azure":
            df = self.azure
        else:
            print("Invalid cloud provider")

        if memory > df.iloc[-1]['GPU RAM']:
            print("No GPU fits your requirements")

        for index, row in df.iterrows():
            if row['GPU RAM'] > memory:
                self.gpu = row[type]
                break

    def get_gpu(self):
        return self.gpu

    def get_cost(self, hours=1):
        return self.gpu * hours


if __name__ == "__main__":
    c = Cost()
    c.find_gpu(5, 'AWS', 'Spot')
    print(c.get_cost(5))
    print(c.gcp['GPU RAM'])
