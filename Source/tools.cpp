#include "tools.h"
#include <random> //For random generation
#include <ctime> //For random generation
#include <sstream> //For istringstream
#include <algorithm> //For copy() and assign()

void TOOLS::Random(const int sigma, const int mi, std::vector<std::vector<double>>& numbers, const double denominator)
{
  //Sigma and mi are ranges([sigma,mi])
  static std::random_device __randomDevice;
  //A Mersenne Twister pseudo-random generator of 32-bit numbers with a state size of 19937 bits.
  static std::mt19937 __randomGen(__randomDevice());
  //Generates random numbers according to the Normal (or Gaussian) random number distribution
  static std::normal_distribution<double> __normalDistribution(0.0, 1.0);
  for(int i = 0; i < sigma; i++)
  {
    std::vector<double> temp;
    for (int j = 0; j < mi; j++) {

      double number = (__normalDistribution(__randomGen)/denominator);
      temp.push_back(number);
    }
    numbers.push_back(temp);
  }
}

std::vector<double> TOOLS::dotProduct(const std::vector<std::vector<double>>& v1, const std::vector<double>& v2)
{
  std::vector<double> v3;
  for (int i = 0; i < v1.size(); i++) {
    double sum = 0;
    for (int j = 0; j < v1[i].size(); j++) {
      sum += v1[i][j]*v2[j];
    }
    v3.push_back(sum);
  }
  return v3;
}

std::vector<std::vector<double>> TOOLS::dotProduct(const std::vector<double>& v1, const std::vector<double>& v2)
{
  std::vector<std::vector<double>> v3;
  for (int i = 0; i < v1.size(); i++) {
    std::vector<double> temp;
    for (int j = 0; j < v2.size(); j++) {
      temp.push_back(v1[i]*v2[j]);
    }
    v3.push_back(temp);
  }
  return v3;
}

void TOOLS::limiting(std::vector<std::vector<double>>& input, const int min, const int max)
{
  for (int i = 0; i < input.size(); i++) {
    for (int j = 0; j < input[0].size(); j++) {
      if (input[i][j] < min)
        input[i][j] = min;
      if(input[i][j] > max)
        input[i][j] = max;
    }
  }
}

void TOOLS::limiting(std::vector<double>& input, const int min, const int max)
{
  for (int i = 0; i < input.size(); i++) {
    if (input[i] < min)
        input[i] = min;
    if(input[i] > max)
        input[i] = max;
  }
}

int TOOLS::randomGen(const int i)
{
  srand(time(0));
  return std::rand()%i;
}

//One-hot vector contains all zeros except for a single one
//The “one” in each one-hot vector will be at the word’s corresponding integer index.
std::vector<std::vector<double>> TOOLS::createInputs(const std::string& text, const std::vector<std::string>& wordsVector)
{
  std::vector<std::vector<double>> oneHotVector;
  std::vector<std::string> words;
  //Used to split string around spaces
  std::istringstream ss(text);
  //Traverse through all words
  do {
      //Read a word
      std::string word;
      ss >> word;
      //Skip if word is empty
      if(word == "")
        continue;
      //Store words into vector(m_keys)(it the word is not exists)
      if (std::find(words.begin(), words.end(), word) == words.end())
        words.push_back(word);
  } while (ss);

  for (int i = 0; i < words.size(); i++) {
    std::vector<std::string>::const_iterator it = std::find(wordsVector.begin(), wordsVector.end(), words[i]);
    int index = std::distance(wordsVector.begin(), it);
    std::vector<double> temp(wordsVector.size(), 0);
    temp[index] = 1;
    oneHotVector.push_back(temp);
  }
  return oneHotVector;
}
