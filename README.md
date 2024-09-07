# fast_word_emotion_analysis

音声の入力を受けとり、音節の切れ目で、それまでの感情を計算します。  
arduinoを使い、感情の計算結果を使って、人の首に適切な量の電流を流し、適切なタイミングに感情に沿った適切なうなづきを、適切に実行させます。
話を聞いてない人や耳が聞こえない人にオススメです。しかし倫理審査が通らなかったため、デモ機はメガネを使った視覚によるうなずき命令装置になります。
文字起こしの精度よりも、適切なうなづきを行うために実行速度を重視しています。理由は聞いていただければ答えます。
Arduino側のコードは……デモ機の中にしか残ってないかも……

動作方法

srcの内部の二つのスクリプトを動作させ、バックグラウンドで動かし続ける。
（二つのスクリプトはfileのioを通じて同期させている。理由はデモ用に支給されたパソコンのスペックがあんまりであったため、処理の負荷を私用の携帯端末に分散させることができるようにするためである。）  
初回は、whisperモデルやbertのモデルのダウンロードが走るため、時間がかかる。  
また、cudaや音声入力の管理が必要なため、環境依存で修正しなければならない諸々が非常に多い。  

Androidのアプリを作りBluetooth経由で動くようにもしたが……忙しすぎて気が狂ってたので、アプリのコードはどっかやってしまった。
