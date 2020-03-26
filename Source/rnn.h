#ifndef __RNN_H__
#define __RNN_H__

#include <vector>
#include <string>

namespace RNN
{
  class Rnn
  {
  public:
    Rnn(const int inputSize, const int outputSize, const int hiddenSize);
    ~Rnn();

    std::vector<double> forward(const std::vector<std::vector<double>>& input);
    void backprop(const std::vector<double>& d_y, const double learn_rate = 0.02);

  private:
    //Hidden states size
    const int m_hiddenSize;

    //Weight vectors
    std::vector<std::vector<double>> m_wHH;
    std::vector<std::vector<double>> m_wXH;
    std::vector<std::vector<double>> m_wHY;
    //Bias vectors
    std::vector<double> m_bH;
    std::vector<double> m_bY;

    //Cached fields
    std::vector<std::vector<double>> m_cachedInput;
    std::vector<std::vector<double>> m_cachedH;

    //Function for initialising weights and biases
    void initWB(const int inputSize, const int outputSize);
    //Softmax activation function
    std::vector<double> softmax(const std::vector<double>& input);
  };
}

#endif
