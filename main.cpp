#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <unistd.h>

using namespace std;

#define VERSION "1.0.0"

vector<pair<string, string>> readCSV(const string& filename) {
	vector<pair<string, string>> dataset;
	ifstream file(filename);
	string line, message, sentiment;

	bool header;
	while (getline(file, line)) {
		stringstream ss(line);
		getline(ss, message, '\t');
		getline(ss, sentiment, '\t');
		if (!header) {
			header = true;
			continue;
		}
		dataset.emplace_back(message, sentiment);
	}

	return dataset;
}

vector<string> tokenize(const string& text) {
	vector<string> tokens;
	string token;
	istringstream tokenStream(text);

	while (tokenStream >> token) {
		tokens.push_back(token);
	}

	return tokens;
}

class NaiveBayes {
private:
	unordered_map<string, int> classCounts;
	unordered_map<string, unordered_map<string, int>> wordCounts;
	unordered_set<string> vocabulary;
	int totalMessages;
	double smoothingFactor;

	void saveModel(const string& filename) {
		ofstream file(filename, ios::binary);
		assert(file.is_open());

		size_t size = classCounts.size();
		file.write(reinterpret_cast<const char*>(&size), sizeof(size));
		for (const auto& [sentiment, count] : classCounts) {
			size = sentiment.size();
			file.write(reinterpret_cast<const char*>(&size), sizeof(size));
			file.write(sentiment.data(), size);
			file.write(reinterpret_cast<const char*>(&count), sizeof(count));
		}

		size = wordCounts.size();
		file.write(reinterpret_cast<const char*>(&size), sizeof(size));
		for (const auto& [sentiment, words] : wordCounts) {
			size = sentiment.size();
			file.write(reinterpret_cast<const char*>(&size), sizeof(size));
			file.write(sentiment.data(), size);

			size = words.size();
			file.write(reinterpret_cast<const char*>(&size), sizeof(size));
			for (const auto& [word, count] : words) {
				size = word.size();
				file.write(reinterpret_cast<const char*>(&size), sizeof(size));
				file.write(word.data(), size);
				file.write(reinterpret_cast<const char*>(&count), sizeof(count));
			}
		}

		size = vocabulary.size();
		file.write(reinterpret_cast<const char*>(&size), sizeof(size));
		for (const auto& word : vocabulary) {
			size = word.size();
			file.write(reinterpret_cast<const char*>(&size), sizeof(size));
			file.write(word.data(), size);
		}

		file.write(reinterpret_cast<const char*>(&totalMessages), sizeof(totalMessages));
		file.write(reinterpret_cast<const char*>(&smoothingFactor), sizeof(smoothingFactor));

		file.close();
	}

	void loadModel(const string& filename) {
		ifstream file(filename, ios::binary);
		assert(file.is_open());

		size_t size;
		file.read(reinterpret_cast<char*>(&size), sizeof(size));
		classCounts.clear();
		for (size_t i = 0; i < size; ++i) {
			string sentiment;
			file.read(reinterpret_cast<char*>(&size), sizeof(size));
			sentiment.resize(size);
			file.read(&sentiment[0], size);
			int count;
			file.read(reinterpret_cast<char*>(&count), sizeof(count));
			classCounts[sentiment] = count;
		}

		size_t wordCountsSize;
		file.read(reinterpret_cast<char*>(&wordCountsSize), sizeof(wordCountsSize));
		wordCounts.clear();
		for (size_t i = 0; i < wordCountsSize; ++i) {
			string sentiment;
			file.read(reinterpret_cast<char*>(&size), sizeof(size));
			sentiment.resize(size);
			file.read(&sentiment[0], size);

			size_t wordsSize;
			file.read(reinterpret_cast<char*>(&wordsSize), sizeof(wordsSize));
			unordered_map<string, int> words;
			for (size_t j = 0; j < wordsSize; ++j) {
				string word;
				file.read(reinterpret_cast<char*>(&size), sizeof(size));
				word.resize(size);
				file.read(&word[0], size);
				int count;
				file.read(reinterpret_cast<char*>(&count), sizeof(count));
				words[word] = count;
			}
			wordCounts[sentiment] = words;
		}
		cout << "Word counts: " << wordCounts.size() << endl;

		size_t vocabSize;
		file.read(reinterpret_cast<char*>(&vocabSize), sizeof(vocabSize));
		vocabulary.clear();
		for (size_t i = 0; i < vocabSize; ++i) {
			string word;
			file.read(reinterpret_cast<char*>(&size), sizeof(size));
			word.resize(size);
			file.read(&word[0], size);
			vocabulary.insert(word);
		}
		cout << "Vocabulary: " << vocabulary.size() << endl;

		file.read(reinterpret_cast<char*>(&totalMessages), sizeof(totalMessages));
		file.read(reinterpret_cast<char*>(&smoothingFactor), sizeof(smoothingFactor));

		file.close();
	}

public:
	NaiveBayes(double smoothing = 1.0) : totalMessages(0), smoothingFactor(smoothing) {}

	void train(const vector<pair<string, string>>& dataset) {
		for (const auto& [message, sentiment] : dataset) {
			classCounts[sentiment]++;
			totalMessages++;
			auto tokens = tokenize(message);

			for (const auto& token : tokens) {
				wordCounts[sentiment][token]++;
				vocabulary.insert(token);
			}
		}
		saveModel("naive_bayes_model.bin");
	}

	string predict(const string& message) {
		auto tokens = tokenize(message);
		unordered_map<string, double> classProbabilities;

		for (const auto& [sentiment, count] : classCounts) {
			classProbabilities[sentiment] = log(static_cast<double>(count) / totalMessages);

			for (const auto& token : tokens) {
				double tokenCount = wordCounts[sentiment][token] + smoothingFactor;  // Add smoothing
				double tokenProbability = tokenCount / (classCounts[sentiment] + vocabulary.size() * smoothingFactor);
				classProbabilities[sentiment] += log(tokenProbability);
			}
		}

		return max_element(classProbabilities.begin(), classProbabilities.end(),
			[](const auto& a, const auto& b) {
				return a.second < b.second;
			})->first;
	}

	void loadExistingModel() {
		loadModel("naive_bayes_model.bin");
	}

private:
	vector<string> tokenize(const string& text) {
		vector<string> tokens;
		string token;
		istringstream tokenStream(text);

		while (tokenStream >> token) {
			tokens.push_back(token);
		}

		return tokens;
	}
};

vector<string> getSplit(string str, string token) {
	vector<string> result;
	while (str.size()) {
		int index = str.find(token);
		if (index != string::npos) {
			result.push_back(str.substr(0, index));
			str = str.substr(index + token.size());
			if (str.size() == 0) result.push_back(str);
		} else {
			result.push_back(str);
			str = "";
		}
	}
	return result;
}

int main(int argc, char** argv) {
	string message = "";
	double smoothingFactor = 1.0;
	if (!isatty(STDIN_FILENO)) {
		string input = "";
		while (getline(cin, input)) {
			message += input + "\n";
		}
		message.pop_back();
	}
	for (int i = 0; i < argc; i++) {
		if ((string)argv[i] == "-h" || (string)argv[i] == "--help") {
			cout << "Usage: " << argv[0] << " [options]" << endl;
			cout << "Options:" << endl;
			cout << "-h, --help:\t\tShow this help message and exit." << endl;
			cout << "--version:\t\tShow the version of the program and exit." << endl;
			cout << "-m, --message:\t\tThe message to predict the sentiment of." << endl;
			cout << "-s, --smoothing:\t\tThe smoothing factor to use for the Naive Bayes model." << endl;
			return 0;
		} else if ((string)argv[i] == "--version") {
			cout << VERSION << endl;
			return 0;
		} else if ((string)argv[i] == "-m" || (string)argv[i] == "--message") {
			i++;
			if (i < argc) {
				message = argv[i];
			} else {
				cerr << "No message provided." << endl;
				return 1;
			}
		} else if ((string)argv[i] == "-s" || (string)argv[i] == "--smoothing") {
			i++;
			if (i < argc) {
				smoothingFactor = stod(argv[i]);
			} else {
				cerr << "No smoothing factor provided." << endl;
				return 1;
			}
		}
	}

	if (message == "") {
		cerr << "No message provided." << endl;
		return 1;
	}

	// try {
	// 	if (ifstream("naive_bayes_model.bin").good() == false) {
	// 		throw runtime_error("Model file does not exist.");
	// 	}
	// 	nb.loadExistingModel();
	// 	cout << "Model loaded successfully." << endl;
	// } catch (const exception& e) {
	// 	cerr << "Failed to load model: " << e.what() << endl;
	// 	string filename = "dataset.csv";
	// 	auto dataset = readCSV(filename);
	// 	nb.train(dataset);
	// }

	NaiveBayes nb(smoothingFactor);
	string filename = "dataset.csv";
	auto dataset = readCSV(filename);
	nb.train(dataset);
	vector<string> messages = getSplit(message, "\n");
	for (const auto& msg : messages) {
		string sentiment = nb.predict(msg);
		cout << "The sentiment of \"" << msg << "\" is: " << sentiment << endl;
	}

	return 0;
}
// TODO: Get the model loading to work.