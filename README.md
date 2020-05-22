# introducing_NN_by_tensorflow
Rを用いた機械学習モデルの構築の練習とそのメモである. 後述のRのpackageを用いて自身のMDシミュレーションを用いた研究に機械学習を導入することを目指す. 正確性を担保するためREADMEは日本語で記す. 同時に英語の練習も兼ねて英語 (README_(English).md) でも記す.  

# Overview    
私の研究は, MD計算を用いて系に電圧を印加した際のカチオンの多孔質カーボンへの選択的吸着の特性を調査するものである. 本試行はRを用いてカチオンの質量 `mass`, 価数 `valent`, 第一/第二水和半径 `r1/r2`, RDFの最大値 `gr_max` 並びに系に印加した電圧 `vol`, 系の細孔径 `pore_d` の7つの特徴量から細孔内へカチオンが吸着される確率 `pred_P` を予測するモデルを作成する. data数は**171個**.         

# Description  
この試行はanaconda3/5.3.1下で行った.  
## Package
* tensorflow 1.12.0 
* pandas 0.25.3  
* numpy 1.14.5  

## constructing NN model
### shaping dataset
7つの特徴量, `mass`, `valent`, `r1`, `r2`, `gr_max`, `vol` and `pore_d`, と1つの被説明変数を`p_total`を dataset.txt から入力.    
このdatasetの**75%** をtraining dataset, 残りをtest datasetとし, それぞれの特徴量を以下の式で正規化した.   
```
normed_train_data = (train_dataset - np.array(norm_min))/np.array(norm_max - np.array(norm_min))
```
`norm_min`と`norm_max`はそれぞれdatasetの最大値, 最小値であり normalization.txt からの入力. (未知datasetの正規化にも対応するためにわざわざこのような形を取った. )  

被説明変数 `p_total` はそのままだと0～1の範囲しか持たないことから誤差が小さく見積もられてしまうので, 100倍して百分率にしてから入力した.

--- memo ---  
`pd.read_csv()`でdataset.txtを読み込むと, dataはobject型となる. このままでは上記の正規化の式に代入してもエラーを吐くので, `train/test_dataset.astype('float')` でfloat型に変換しておく必要がある.   


### definition NN model
今回はとりあえず以下のNNモデルを構築した. ハイパーパラメータはこれからチューニングする予定.  
今回は参考にさせて頂いたHPのやり方 <a href="https://qiita.com/SwitchBlade/items/6677c283b2402d060cd0" target="_blank">[here]</a>に従い, 別ファイル, NN_model.py, にclassを作ってNNモデルを定義した.  

![NN model](https://github.com/yusuke-kawaian/introducing_NN_by_tensorflow/blob/master/DNN1.png)


### loss function
今回は**RMSE**を損失関数とし, **Adam Optimizer**で最適化した. 同時に**MSA**も計算しTensorBoardで追跡しておいた.  
ちなみに, 精度の評価について参考にしたのはこちら. <a href="https://pythondatascience.plavox.info/scikit-learn/回帰モデルの評価" target="_blank">[here]</a>  
    
``` 
def loss1(y, y_):
    with tf.name_scope("calculate_RMSE") as scope:
       #loss = -tf.reduce_sum(labels*tf.log(logits)) #closs entropy
       #loss = tf.reduce_mean(tf.square(y - y_))
       rmse = tf.sqrt(tf.losses.mean_squared_error(labels = y_, predictions = y))
       return rmse

def loss2(y, y_):
    with tf.name_scope("calculate_MAE") as scope:
       mae = tf.reduce_mean(tf.abs(y_ - y))
       return mae

def training(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step
```

### accurracy  
このaccuracyが一番頭を悩ませた部分であった. 以前述べたように以下のコードを用いるとこの回帰問題ではずっと1を示していた.  
```
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))
```  
よくよくこのコードについて勉強する <a href="http://testpy.hatenablog.com/entry/2016/11/27/035033" target="_blank">[here]</a> と, 値の合否をTRUE/FALSE判定して, その1/0を用いて計算していた. つまり, **回帰問題には不適であったということである.** (ネットで調べてもみんなMNISTの手書き数字の判定ばかりだからなかなか気づかなかった…)  
結論としては, **本試行ではRMSEとMSAをaccuracyの指標として使用した.** 近いうちに決定係数<img src="https://latex.codecogs.com/gif.latex?R^2"/>を導入したい.  

### training model
上述のモデルを用いてバッチ学習を **1000000step** 行った.  計算には京都大学のスーパーコンピュータシステム laurelを使用した.  

--- memo ---  
laurelでは以下のコマンドで NN_traian.py を実行した.  
```
module load anaconda 3/5.3.1  
tssrun python3 NN_train.py -W hh:mm
```  

### Output glaphs by TensorBoard  
今回は`tf.name_scope()`や`tf.summary()`関数を用いてTensorBoardでモデルを描画した.  
TensorBoardはanaconda上で仮想環境を構築したクライアントPCで起動した. (`tensorboard --logdir=[TensorBoardディレクトリの絶対パス]`で起動後, **http://localhost:6006** で開く. )  

![NN model](https://github.com/yusuke-kawaian/introducing_NN_by_tensorflow/blob/master/DNN1_tensorboard.png)


# Conclusion    
~~tensorflowを用いてNNモデルを作成することは成功したが、以下の結果のように**lossは収束するもののaccuracyがずっと1を示す事象を観測した**. このNN modelは以前別の被説明変数に対して作ったものと同様であるにも関わらず、今回被説明変数を`pre_p`に変更したことで以下の挙動をしめすようになった.~~  

```
step 0, training accuracy 1, loss 201.881
step 1000, training accuracy 1, loss 14.555
step 2000, training accuracy 1, loss 11.1428
step 3000, training accuracy 1, loss 7.86557
step 4000, training accuracy 1, loss 5.43377
step 5000, training accuracy 1, loss 4.0193
step 6000, training accuracy 1, loss 3.29407
step 7000, training accuracy 1, loss 3.09101
step 8000, training accuracy 1, loss 3.04178
step 9000, training accuracy 1, loss 3.00617
step 10000, training accuracy 1, loss 2.92083
```  
accuracyの項で述べたように以前のaccuracyの定義が回帰問題にそぐわなかったと考えられる. 

![RMSE](https://github.com/yusuke-kawaian/introducing_NN_by_tensorflow/blob/master/RMSE_train_20200521_1Mstep.png)  

# My Problems  
* ~~lossは収束するが, accuracyがずっと1を示している.~~　→　そもそも回帰問題に以前の定義のaccuracyとして用いることがナンセンスであると考えられる. 回帰問題は現在lossとして用いているMSE, その他にはRMSE, MSA, R2値を用いるのが良いと考えられる.   
* ~~今回の被説明変数 `pred_P` は確率の値である. このmodelの出力層はsoftmax関数を通すべきか. また, error fuc. にはMSEとcloss enthoropyどちらが適切か.~~ →　softmax関数は一般的にクラス分類で用いられている活性化関数である. この関数の概形を見ると0付近と1付近の値を取りやすく設計してあることが分かる. 今回はあくまで回帰問題であるため, linear-outputが妥当であると判断する.   
