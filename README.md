#株価予測モデル　

これはRNNの一種であるLSTMを使用して株価を予測するプログラムです。

##概要
過去の株価のデータ(終値、始値、高値、出来高など)のデータをLSTMに入力して入力データの最後のデータの次の日の終値を予測するプログラムです。CSVファイルのパス、バッチサイズ、エポック数は実行後にキーボード入力にで指定する。

##使用技術
-Python
-Pandas
-Numpy
-TensorFlow / Keras(LSTM)
-Scikit-learn
-LightGBM

##使い方
1. 仮想環境を構築し、ライブラリをインストールします。
    pip install -r requirements.txt
2. スクリプトを実行する。
    python data_cont.py