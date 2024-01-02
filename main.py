from ToolsContainer import preprocess, Webscraper
import pandas as pd
import os
from datetime import date
import datasets
import yfinance as yf
import pandas as pd
import os


# Legend:
# 2 is positive
# 1 neutral
# 0 is negative

path = os.getcwd()

# CONFIGURATION :sentences_50agree; Number of instances with >=50% annotator agreement: 4846 (compliant with paper2)
dataset = datasets.load_dataset('financial_phrasebank', name = 'sentences_50agree')

print(dataset)

df = dataset.data['train'].to_pandas() # all rows are contained inside the 'train' set (no 'test')
df.to_csv(path_or_buf= path + '/Datasets/Input/financial_phrasebank.csv')

pass

HEAD = "https://www.marketwatch.com/investing/stock/"
TICKER = "baba"
TAIL = "?mod=mw_quote_tab"


father_tag = "div"
father_attr = "class"
father_attr_name = "column column--primary j-moreHeadlineWrapper"
headline_tag = "a"
headline_attr = "class"
headline_attr_name = "link"
time_tag = "span"
time_attr = "class"
time_attr_name = "article__timestamp"
author_tag = "span"
author_attr = "class"
author_attr_name = "article__author"
n_headlines = 1200
scroll_step = 200


df_MW = Webscraper.NewScraper(TICKER, HEAD, father_tag, father_attr, father_attr_name, headline_tag, headline_attr,
                   headline_attr_name, time_tag, time_attr, time_attr_name, author_tag, author_attr, author_attr_name, scroll_step, n_headlines, TAIL)

df_MW['timestamp'] = Webscraper.Timestamps_Cleaning(df_MW['timestamp'])

df_MW.set_index('timestamp', inplace= True)

df_MW.to_csv(path + '/GPU_Server_Run/Input_Data/MWscraping_{data}.csv'.format(data = str(date.today())))

# df_MW_groupby = df_MW.groupby('author').count()
# df_MW_groupby.to_csv(path + '/Input/MWscraping_groupby_{data}.csv'.format(data = str(date.today())))



path_subtask1 = path + '/Datasets/Input/semeval-2017-task-5-subtask-1/'
path_subtask2 = path + '/Datasets/Input/semeval-2017-task-5-subtask-2/'


df = pd.read_csv(path + '/Datasets/Input/financial_phrasebank.csv', index_col= 0)


df2_train = preprocess.json_to_pandas_converted(path_subtask1 + 'Microblog_Trainingdata.json')[['spans','label']]
df2_trial = preprocess.json_to_pandas_converted(path_subtask1 + 'Microblog_Trialdata.json')[['spans','label']]
df2_train.columns = ['sentence','label']
df2_trial.columns = ['sentence','label']

semeval_subtask1 = pd.concat([df2_train, df2_trial], axis = 0)
semeval_subtask1.to_csv(path + '/Datasets/Input/semeval_subtask1.csv')

df3_train = preprocess.json_to_pandas_converted(path_subtask2 + 'Headline_Trainingdata.json')[['title','label']]
df3_trial = preprocess.json_to_pandas_converted(path_subtask2 + 'Headline_Trialdata.json')[['title','label']]
df3_train.columns = ['sentence','label']
df3_trial.columns = ['sentence','label']

semeval_subtask2 = pd.concat([df3_train, df3_trial], axis = 0)
semeval_subtask2.to_csv(path + '/Datasets/Input/semeval_subtask2.csv')

df4 = pd.concat([df,df2_train,df2_trial,df3_train,df3_trial], axis = 0, ignore_index= True)
df4 = preprocess.WrapFunction(df4, col_name= 'sentence')
df4.to_csv(path + '/GPU_Server_Run/Input_Data/Final_Dataset.csv')

df5 = pd.read_csv(path + '/Datasets/Input/MWscraping_2023-11-03.csv', index_col= 0)
df5 = preprocess.WrapFunction(df5, col_name= 'sentence')
df5.to_csv(path + '/GPU_Server_Run/Input_Data/WS_Final_Dataset.csv')


pass


WS = pd.read_csv(path + "/Datasets/Input/MWscraping_2023-11-03.csv", index_col= 0)

TICKER = ['BABA']
lower_bound_time = pd.Timestamp(WS.index[-1])
upper_bound_time = pd.Timestamp(WS.index[0])

# Get the data for the stock BABA
data = yf.download(TICKER[0],lower_bound_time, upper_bound_time)

data.to_csv(path + f'/GPU_Server_Run/Input_Data/{TICKER[0]}_financial_data.csv')

pass

