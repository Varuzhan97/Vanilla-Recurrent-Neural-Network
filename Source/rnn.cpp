#include <iostream>
#include <iterator> //For back_inserter
#include <cmath> //For tanh()

#include "rnn.h"
#include "tools.h"

RNN::Rnn::Rnn(const int inputSize, const int outputSize, const int hiddenSize) : m_hiddenSize(hiddenSize)
{
  //Weights and Biases initialization
  initWB(inputSize, outputSize);
}

RNN::Rnn::~Rnn()
{
}

//Weights initialization
void RNN::Rnn::initWB(const int inputSize, const int outputSize)
{
  //Diving by 1000 during the initialization is important.
  //If the initial values are too large or too small, training the network will be ineffective
  TOOLS::Random(m_hiddenSize, m_hiddenSize, m_wHH, 1000.0);
  TOOLS::Random(m_hiddenSize, inputSize, m_wXH, 1000.0);
  TOOLS::Random(outputSize, m_hiddenSize, m_wHY, 1000.0);

  //Biases initialization
  m_bH.assign(m_hiddenSize, 0.0);
  m_bY.assign(outputSize, 0.0);
}

std::vector<double> RNN::Rnn::forward(const std::vector<std::vector<double>>& input)
{
  //Clear last cached input and make cache
  m_cachedInput.clear();
  m_cachedInput = input;

  std::vector<double> h(64, 0.0);

  //Clear last cached h and make cache
  m_cachedH.clear();
  m_cachedH.push_back(h);

  std::vector<double> y(2, 0.0);

  //Formula -- (h = tanh(m_wXH X input[i] + m_wHH X h + m_Bh)
  //Iterating over every one-hot vector
  for (int i = 0; i < input.size(); i++) {
    std::vector<double> element = input[i];
    //m_wXH X input[i]
    std::vector<double> dotProduct1 = TOOLS::dotProduct(m_wXH, element);
    //m_wHH X h
    std::vector<double> dotProduct2 = TOOLS::dotProduct(m_wHH, h);
    //Sum all, count tanh and assign to h
    for (int k = 0; k < h.size(); k++) {
      h[k] = tanh(dotProduct1[k]+dotProduct2[k]+m_bH[k]);
    }
    //Cache next h
    m_cachedH.push_back(h);
  }

  //Formula -- (y = m_wHY X h + m_bY)
  std::vector<double> dotProduct = TOOLS::dotProduct(m_wHY, h);
  for (int k = 0; k < y.size(); k++) {
    y[k] = dotProduct[k]+m_bY[k];
  }
  std::vector<double> predictions = softmax(y);
  return predictions;
}

std::vector<double> RNN::Rnn::softmax(const std::vector<double>& input)
{
  //Formula -- y = exp(input) / sum(exp(input))
  double exp_sum = 0.0;

  std::vector<double> exponential;
  std::vector<double> predictions;

  //counts "exp(input)" and "sum(exp(input))" parts
  for(int i = 0; i < input.size(); i++)
  {
    double temp = exp(input[i]);
    exp_sum+=(temp);
    exponential.push_back(temp);
  }

  //Division part
  for(int i = 0; i < input.size(); i++)
  {
    double temp = ((double)exponential[i]/(double)exp_sum);
    predictions.push_back(temp);
  }
  return predictions;
}

void RNN::Rnn::backprop(const std::vector<double>& d_y, const double learn_rate)
{
  //Perform a backward pass of the RNN
  int n = m_cachedInput.size();
  //Calculate dL/dWhy and dL/dby.
  std::vector<double> tempH = m_cachedH[n];
  std::vector<std::vector<double>> d_wHY = TOOLS::dotProduct(d_y, tempH);
  std::vector<double> d_bY(d_y);

  //Initialize dL/dWhh, dL/dWxh, and dL/dbh to zero.
  std::vector<std::vector<double>> d_wHH;
  for (int i = 0; i < m_wHH.size(); i++) {
    std::vector<double> zeros(m_wHH[0].size(),0);
    d_wHH.push_back(zeros);
  }
  std::vector<std::vector<double>> d_wXH;
  for (int i = 0; i < m_wXH.size(); i++) {
    std::vector<double> zeros(m_wXH[0].size(), 0);
    d_wXH.push_back(zeros);
  }
  std::vector<double> d_bH(64, 0);

  //Calculate dL/dh for the last h.
  //Before calculation we must transpose
  std::vector<std::vector<double>> temp_wHY;
  for (int i = 0; i < m_wHY[0].size(); i++) {
    std::vector<double> zeros(m_wHY.size(), 0);
    temp_wHY.push_back(zeros);
  }
  for(int i = 0; i < m_wHY.size(); ++i)
  {
    for(int j = 0; j < m_wHY[0].size(); ++j)
    {
      temp_wHY[j][i]=m_wHY[i][j];
    }
  }
  std::vector<double> d_h = TOOLS::dotProduct(temp_wHY, d_y);

  //Backpropagate through time.
  for (int i = 0; i < n; i++) {
    //An intermediate value: dL/dh * (1 - h^2)
    std::vector<double> temp;
    for (int j = 0; j < m_cachedH[n-i].size(); j++) {
      double value = pow(m_cachedH[n-i][j], 2);
      temp.push_back((1 - value)*d_h[j]);
    }
    //dL/db = dL/dh * (1 - h^2)
    for (int j = 0; j < d_bH.size(); j++) {
      d_bH[j] += temp[j];
    }
    //dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
    std::vector<std::vector<double>> temp_d_wHH = TOOLS::dotProduct(temp, m_cachedH[n-i-1]);
    //dL/dWxh = dL/dh * (1 - h^2) * x
    std::vector<std::vector<double>> temp_d_wXH = TOOLS::dotProduct(temp, m_cachedInput[n-i-1]);
    for (int j = 0; j < d_wHH.size(); j++) {
      for (int k = 0; k < d_wHH[0].size(); k++) {
        d_wHH[j][k] += temp_d_wHH[j][k];
      }
    }
    for (int j = 0; j < d_wXH.size(); j++) {
      for (int k = 0; k < d_wXH[0].size(); k++) {
        d_wXH[j][k] += temp_d_wXH[j][k];
      }
    }
    //dL/dh = dL/dh * (1 - h^2) * Whh
    d_h.clear();
    d_h = TOOLS::dotProduct(m_wHH, temp);
  }

  //Clip gradient values that are below -1 or above 1.
  //This helps mitigate the exploding gradient problem,
  //which is when gradients become very large due to having lots of multiplied terms
  //Exploding or vanishing gradients are quite problematic for vanilla RNNs.
  TOOLS::limiting(d_wXH);
  TOOLS::limiting(d_wHH);
  TOOLS::limiting(d_wHY);
  TOOLS::limiting(d_bH);
  TOOLS::limiting(d_bY);

  //Weights update
  for (int i = 0; i < m_wHH.size(); i++) {
    for (int j = 0; j < m_wHH[0].size(); j++) {
      m_wHH[i][j] -= (learn_rate * d_wHH[i][j]);
    }
  }
  for (int i = 0; i < m_wXH.size(); i++) {
    for (int j = 0; j < m_wXH[0].size(); j++) {
      m_wXH[i][j] -= (learn_rate * d_wXH[i][j]);
    }
  }
  for (int i = 0; i < m_wHY.size(); i++) {
    for (int j = 0; j < m_wHY[0].size(); j++) {
      m_wHY[i][j] -= (learn_rate * d_wHY[i][j]);
    }
  }

  //Biases update
  for (int i = 0; i < m_bH.size(); i++) {
    m_bH[i] -= (learn_rate * d_bH[i]);
  }
  for (int i = 0; i < m_bY.size(); i++) {
    m_bY[i] -= (learn_rate * d_bY[i]);
  }
}
