# はじめに

[「ゼロから作るDeep Learning ――Pythonで学ぶディープラーニングの理論と実装」](https://www.oreilly.co.jp/books/9784873117584/)
(オライリー・ジャパン 斎藤 康毅　著)はDeep Learningを初歩から解説した本です。プログラミング言語としてPythonを使用していますが、
本投稿は（およびこれに続く一連の投稿では）これをJavaで実装します。ただし
[ゼロから作るDeep Learning](https://www.oreilly.co.jp/books/9784873117584/)
を置き換えるものではなく、併読することを前提として記述しています。

# 使用する外部ライブラリ

## Numpy


[ゼロから作るDeep Learning](https://www.oreilly.co.jp/books/9784873117584/)
では数値計算のためのライブラリとして
[NumPy](http://www.numpy.org/)
を使用しています。本投稿では
[ND4J](https://nd4j.org/index.html)
を使用します。ND4JはNumPyやMatlibに似た機能を持つJava用のライブラリです。オープンソースの
[Deeplearning4J](https://deeplearning4j.org/)
で使用されていることで有名です。ただしドキュメントはそれほど整備されていません。多少、試行錯誤して使い方を習得する必要があります。
ND4Jは
[CUDA](https://ja.wikipedia.org/wiki/CUDA)
を経由してGPUを使用することができます。高速化するためには有利であると考えます。

## Matplotlib

[ゼロから作るDeep Learning](https://www.oreilly.co.jp/books/9784873117584/)
ではグラフ描画のためのライブラリとして
[Matplotlib](https://matplotlib.org/)
を使用しています。Deep Learningを実現する上でグラフは必須ではないため、本投稿ではグラフ描画のためのライブラリは使用しません。

# 進め方


[ゼロから作るDeep Learning](https://www.oreilly.co.jp/books/9784873117584/)
に登場するPythonで記述されたプログラムを逐次Javaで書き直していきます。結果が同じになることを確認するため、
JUnitのコードとして記述します。例えば以下のような感じです。

```java
public class C1_5_NumPy {

    @Test
    public void C1_5_2_NumPy配列の生成() {
        INDArray x = Nd4j.create(new double[] {1.0, 2.0, 3.0});
        assertEquals("[1.00,2.00,3.00]", Util.string(x));
    }
}
```

これは「第1章 Python入門」の「1.5.2 NumPy配列の生成」に登場するサンプルプログラムをJavaで記述したものです。クラス名やメソッド名に日本語を使用しているので、環境によっては動作しないかもしれません。私の環境（Windows10 + Eclipse Oxygen.2 Release 4.7.2）では問題なく動きます。
Pythonのコードが記述されている節についてのみ記述しているので見出しの項番は連番になっていません。
Javaで記述したすべてのプログラムはGitHubの
[saka1029/Deep.Learning](https://github.com/saka1029/Deep.Learning)
に掲載しています。

# 環境設定

GitHub上のプロジェクトをビルドするためには以下の依存関係をpom.xmlに設定する必要があります。

```xml

  <dependencies>
  	<dependency>
  		<groupId>org.nd4j</groupId>
  		<artifactId>nd4j-native-platform</artifactId>
  		<version>0.9.1</version>
  	</dependency>
  	<dependency>
		<groupId>org.slf4j</groupId>
		<artifactId>slf4j-log4j12</artifactId>
		<version>1.7.2</version>
	</dependency>
  </dependencies>
```
またIDE上のプロジェクトの設定で*Java8*を使用するように設定しておく必要があります。Java9とすると実行時にクラスローダが例外をスローすることがあります。
