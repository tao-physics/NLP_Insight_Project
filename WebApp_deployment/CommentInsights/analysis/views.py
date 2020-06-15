from django.shortcuts import render
from django.http import HttpResponse

from model_roberta_classifier_label_data_aug.src.inference import predict_topic
from model_roberta_classifier_label_data_aug.src import config

from model_roberta_classifier_sentiment.src.inference import predict_sentiment
from model_roberta_classifier_sentiment.src import config as config_s

from text_cleaning.cleantx import text_prepare

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

###########################
from wordcloud import WordCloud, STOPWORDS 
stopwords = set(STOPWORDS) 
###########################

for_prediction_path = './static/data/for_prediction.csv'
input_topic_path = for_prediction_path
output_topic_path = './static/data/topic_done.csv'
input_sentiment_path = output_topic_path
output_final_path = './static/download_result/topic_sentiment_done.csv'
output_final_clean_path = './static/download_result/topic_sentiment_clean.csv'
fig_path = './static/download_result/topic_sentiment_hist.jpg'
fig_wc_path = './static/download_result/topic_sentiment_wc.jpg'

# Create your views here.

def homepage(request):
    return render(request, 'homepage.html')


def upload_file(request):
    if request.method == 'GET':
        return render(request, 'upload.html')
    elif request.method == 'POST':
        doc = request.FILES.get("file")

        print(type(doc))

        with open('./static/data/for_prediction.csv', 'wb') as save_file:
            try:
                for part in doc.chunks():
                    save_file.write(part)
                    save_file.flush()
            except:
                return render(request, 'upload_error.html')

        predict_topic(input_topic_path, output_topic_path)

        predict_sentiment(output_topic_path, output_final_path)

        final_file = pd.read_csv(output_final_path)

        topic_dict = {0: 'Appreciation/Recognition',
            1: 'Balancing Personal/Productivity',
            2: 'Benefits',
            3: 'Co-Workers/Teamwork',
            4: 'Communication/Support',
            5: 'Employee relations',
            6: 'Incentives/Growth',
            7: 'Inspiring Work/Place',
            8: 'Learning & Development',
            9: 'Management Reliability/Integrity',
            10: 'No answer/Nothing',
            11: 'Other',
            12: 'Staffing and Scheduling',
            13: 'Supplies/Systems'}

        final_file['topic_1final'] = final_file['topic_1stpred'].replace(
                topic_dict
            )

        final_file['topic_2final'] = final_file['topic_2ndpred'].replace(
                topic_dict
            )

        final_file['topic_2nd_ratio'] = final_file['prob_topic_2rank'] / (1 - final_file['prob_topic_1rank'])

        final_file['sentiment_1final'] = final_file['sentiment_pred'].replace(
            {0: 'Negative', 1: 'Nothing', 2: 'Positive'}
            )

        final_file['sentiment_2final'] = final_file['sentiment_2ndpred'].replace(
            {0: 'Negative', 1: 'Nothing', 2: 'Positive'}
            )

        final_clean = final_file[['Comment', 'topic_1final', 'topic_2final', 'topic_2nd_ratio', 'sentiment_1final', 'sentiment_2final']]

        final_clean.to_csv(output_final_clean_path, index=False)

        plt.figure(figsize=(20,15))
        ax = sns.countplot(y='topic_1final', data=final_clean, hue='sentiment_1final', palette={'Positive':'C0','Negative':'r','Nothing':(0.8,0.8,0.8)})
        ax.set_ylabel('Topic', fontsize=30)
        ax.set_xlabel('Count', fontsize=30)
        ax.legend(fontsize=25, loc='center right')
        ax.tick_params(axis='x', labelsize=25)
        ax.tick_params(axis='y', labelsize=25)

        plt.savefig(fig_path, bbox_inches='tight')

        ###################################################################################

        final_clean['Comment_cleaned'] = final_clean['Comment'].apply(text_prepare)
        

        nrow = 3
        ncol = 5
        fig, ax = plt.subplots(nrow, ncol, figsize=(35, 21))
        nlabel = 14

        for k in range(5*3):
            
            if k < nlabel:
            
                comment_words = final_clean[final_clean['topic_1final'] == topic_dict[k]]['Comment_cleaned'].values
                comment_words = ' '.join(comment_words)

                wordcloud = WordCloud(width = 800, height = 800, 
                                background_color ='white', 
                                stopwords = stopwords, 
                                min_font_size = 10).generate(comment_words) 

                # plot the WordCloud image
                row = k // ncol
                column = k % ncol

                ax[row, column].imshow(wordcloud) 
                ax[row, column].axis("off")
                ax[row, column].set_title(topic_dict[k], fontsize=30)
            
            else:
                row = k // ncol
                column = k % ncol
                ax[row, column].axis("off") 
                
        plt.tight_layout(pad = 0) 
        fig.savefig(fig_wc_path, bbox_inches='tight') 


        return render(request, 'upload_result.html')


def prediction_result(request):
    return render(request, 'upload_result.html')


#####################################################################

