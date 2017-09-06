# coding: utf-8
"""
まだやっていないこと
1, 形態素解析失敗時の修正
    ex) '0歳'を'0'と'歳'に分けてしまう
2, 自然言語処理の細々した処理
    ex) 'YouTube'と'youtube'を同一のものとしていない，かも？？
3対応済, Stopword決め (tfidf作る時に無視されるのでこちらでは対応する必要がない)
4対応済, tfidfの次元削減
5, 並列化
6, 高速化
7, データベースへのコネクトエラーへの対処
"""
import sys
import MeCab
import pandas as pd
import numpy as np
import mysql.connector ## mysql-connector-python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle


def morpho_1sentence_to_naiyougo(text):
    """
    形態素解析を行う関数
    text: string型の文
    """
    mecab = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")


    words = [] ## 内容語(word)のリスト
    for chunk in mecab.parse( text ).splitlines()[:-1]:
        ## chunk : 形態素解析の結果．以下のようになっている
        ## 広告代理店\t名詞,固有名詞,一般,*,*,*,広告代理店,コウコクダイリテン,コーコクダイリテン
        ## EOSを省くために-1している
        (surface, feature) = chunk.split("\t") ## tabで分割
        if feature.startswith("名詞") or feature.startswith("形容詞")\
        or feature.startswith("動詞") or feature.startswith("助動詞")\
        or feature.startswith("副詞"):
            feature_elems = feature.split(',') ## featureを','で分割してリストに
            words.append(feature_elems[6]) ## 基本形
    return words

## 類似度計算 ---------------------------start
def cosine_similarity_value(v1, v2):
    """
    ベクトルv1, v2のcos類似度の算出
    """
    return sum([a*b for a, b in zip(v1, v2)])/(sum(map(lambda x: x*x, v1))**0.5 * sum(map(lambda x: x*x, v2))**0.5)

def cosine_similarity_vec(v1, mat2): ## あるベクトルと行列の類似度計算
    """
    v1  : numpyのarray型vector
    mat : numpyのarray型matrix
    """
    cosvec = np.empty( mat2.shape[0] )
    for i, v2 in enumerate(mat2):
        # print(cosine_similarity_value(v1, v2))  
        # print(i)
        cosvec[i] = cosine_similarity_value(v1, v2)
    return cosvec

def cosine_similarity_mat(mat): ## 類似度行列を求める． 並列計算できる
    """
    mat : numpyのarray matrix
    output : 類似度行列
    """
    simmat = np.empty( (mat.shape[0], mat.shape[0]) ) ## 行の数
    for i, m in enumerate(mat):
        # print(cosine_similarity_vec( m , mat))
        simmat[i, ] = cosine_similarity_vec( m , mat)
    return simmat
## 類似度計算 --------------------------- end


if __name__=='__main__':
    version = sys.argv[1]
    #mypath = '/root/inoue/nlp/' + version + '/'
    #mypath = '/var/www/html/lbcb/'
    mypath = '/var/www/html/lbcb/' + version + '/'
    question = sys.argv[2]
    # question = '　おもしろ動画やどっきり動画をつくって稼ぎたい場合、どんな方法がベストだと思いますか？　YouTubeで広告収入で稼ぐか月額制とかで有料で公開して稼ぐとかいろいろあると思いますがアドバイスください。'

    ##
    if "今年もあと" in question:
        print("堀江さんが今年一番楽しかったことって何ですか？")
        print("そりゃ、たくさんあったよ。主にHIUで。ここクリックしてみ？＜URL＞")
        sys.exit()


    ## 形態素解析
    word_list = morpho_1sentence_to_naiyougo(question)
    joinword = ' '.join(word_list) ## 半角空白を入れて連結
    #print(joinword)
    
    ## tdidf計算
    ### tdidf計算オブジェクトをunpickle
    with open(mypath + 'vectorizer.pickle', mode='rb') as f:
        vectorizer_ = pickle.load(f)
    with open(mypath + 'transformer.pickle', mode='rb') as f:
        transformer_ = pickle.load(f)
    ### vectorizer_に基づきtfベクトルを作成(新語は無視)
    tf_obj_new = vectorizer_.transform([joinword])
    # print(tf_obj_new)
    ### transformer_によってtfidfベクトルを計算
    tfidf_obj_new = transformer_.transform(tf_obj_new)
    # print(tfidf_obj_new)
    if sum(tfidf_obj_new.toarray()[0]) == 0.0: ## 過去の単語になかった場合
        print("過去にそんな質問はなかったよ")
        sys.exit()

    ## LSI
    with open(mypath + 'svd0_fitted.pickle', mode='rb') as f:
        svd0_fitted = pickle.load(f)
    tfidf_trans_new = svd0_fitted.transform(tfidf_obj_new)[0]
    
    ## 類似度計算
    ### 過去のSVD済みtfidfベクトルを保存したcsvをpd.DataFrameとして読み込み
    df_old_ = pd.read_csv(mypath + "res_tfidf_mat.csv", index_col='id')
    ### numpyのarrayの行列に変換
    mat_old_ = df_old_.as_matrix()
    ### 過去の文章全てと計算して類似度のベクトルを返す
    result_similarity = cosine_similarity_vec(tfidf_trans_new, mat_old_)
    ### 結果をDataFrameに
    df_res_0 = pd.DataFrame( { 'id':df_old_.index, 'sim':result_similarity } )
    df_res_ = df_res_0.set_index('id') ## indexを'id'でset
    # print(df_res_)

    ## 計算結果をsortして類似度高いidを取り出す
    #k = 10 ## 何個取り出すか
    k = 1
    df_res_sorted = df_res_.sort_values(by='sim', ascending=False)
    id_top = df_res_sorted[0:k].index
    # print(id_top)
    
    ## データベースへ接続 (コネクトエラー考慮してない)
    connect = mysql.connector.connect(
                user='h_ai_dbuser',
                password='h_ai_dbuser_pw_20160831',
                host = 'haidatabase.caejmrqyinlc.ap-northeast-1.rds.amazonaws.com',
                database='hAiDatabase',
                charset='utf8'
                )
    
    ## 上位から順にQ&Aを取り出していく
    QA_list = []
    for n, id in enumerate(id_top):
        sql_text = 'select * from qa_data6 where id = {0}'.format(id)
        df_ = pd.io.sql.read_sql(sql_text, connect) ## DFとして取り出す
        df_tmp = df_.set_index('id') ## indexにid入れておかないと何故か次の行でうまくテキスト取り出せない
        ## questionとanswer両方とも用いる．
        #QA_list.append("{0}番目に近い質問\n{1}\n\n回答\n{2}\n".format(n+1,df_tmp.ix[id, 'question'], df_tmp.ix[id, 'answer']) )
    
        df_tmp.ix[id,'question'] = df_tmp.ix[id,'question'].replace('\n','')
        df_tmp.ix[id,'answer'] = df_tmp.ix[id,'answer'].replace('\n','')
        print(df_tmp.ix[id,'question'])
        print(df_tmp.ix[id,'answer'])
 
        #適当にjoin
    #response_text = '--------------------\n\n'.join(QA_list)
    #print(response_text)
