<p align="center">
	<img src="./asset/logo.png" alt="Sentiment Sentinel Logo" width="200" height="100">
</p>

<h1 align="center">Sentiment Sentinel</h1>

<p align="center">
	<strong>Predict the sentiment of messages using AI!</strong>
</p>

## 🚀 Overview

Welcome to **Sentiment Sentinel**! The program takes in a message and predicts the sentiment of the message using machine learning (ML). The program uses the Naive Bayes classifier, which is a supervised ML algorithm.

> [!NOTE]
> The following command will show the manual for the program: `man ./sentiment_sentinel.1`.

## 🎨 Features

- **AI:** Predict the sentiment of messages using ML.
- **Input:** Take in input from piped commands or command line arguments.

> [!WARNING]
> The program currently cannot store a local model binary on the disk and read from it, but it is in the works.

## 🛠️ Installation

To get started with the program, follow the steps below:

1. **Clone the Repository**
```sh
git clone https://github.com/321BadgerCode/sentiment_sentinel.git
cd ./sentiment_sentinel/
```

2. **Compile the Program**
```sh
g++ ./main.cpp -o ./sentiment_sentinel
```

## 📈 Usage

To use the program, there is only **one** step!

1. **Run the program**
```sh
./sentiment_sentinel [options]
# This will run the program with a bunch of test messages.
# cat ./test.txt | ./sentiment_sentinel
```

<details>

<summary>💻 Command Line Arguments</summary>

**Command Line Arguments**:
|	**Argument**		|	**Description**		|	**Default**	|
|	:---:			|	:---:			|	:---:		|
|	`-h & --help`		|	Help menu		|			|
|	`--version`		|	Version number		|			|
|	`-m & --message`	|	Message to predict	|			|
|	`-s & --smoothing`	|	Smoothing factor	|	1.0		|

</details>

## 📜 License

[LICENSE](./LICENSE)