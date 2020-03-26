#ifndef __TOOLS_H__
#define __TOOLS_H__

#include <vector>
#include <string>

namespace TOOLS
{
  //Returns normally(Gaussian) distributed random number
  //The last parameter is for division(for reducing the initial variance)
  void Random(const int sigma, const int mi, std::vector<std::vector<double>>& numbers, const double denominator = 1.0);

  //Matrix multiplication function
  std::vector<double> dotProduct(const std::vector<std::vector<double>>& v1, const std::vector<double>& v2);
  std::vector<std::vector<double>> dotProduct(const std::vector<double>& v1, const std::vector<double>& v2);

  //In-place clipping functions
  //Given an interval, values outside the interval are clipped to the interval edges.
  //For example, if an interval of [0, 1] is specified, values smaller than 0 become 0,
  //and values larger than 1 become 1.
  void limiting(std::vector<std::vector<double>>& input, const int min = -1, const int max = 1);
  void limiting(std::vector<double>& input, const int min = -1, const int max = 1);

  //Random generator
  int randomGen(const int i);

  //Function for creating input matrix for RNN
  //Returns a vector of one-hot vectors representing the words in the input text string.
  std::vector<std::vector<double>> createInputs(const std::string& text, const std::vector<std::string>& wordsVector);
}

#endif
