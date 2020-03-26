#include <iostream>
#include <algorithm> // For max_element() and random_shuffle()
#include <cmath> //For log()
#include "tools.h"
#include "reader.h"
#include "rnn.h"



void process(const std::map<std::string, std::string>& items, const std::vector<std::string>& vocabulary, RNN::Rnn * rnn, double& loss, double& accuracy, bool backprop = true)
{
  loss = 0.0;
  accuracy = 0.0;
  
  //Vector for random shuffling items
  std::vector<std::string> shuffledItems;
  for (std::map<std::string, std::string>::const_iterator itr = items.begin(); itr != items.end(); ++itr)
  {
    shuffledItems.push_back(itr->first);
  }
  //Shuffling items of vector
  random_shuffle(shuffledItems.begin(), shuffledItems.end(), TOOLS::randomGen);

  for (int i = 0; i < shuffledItems.size(); i++)
  {
    //Generate one-hot vector for every shuffled item
    std::vector<std::vector<double>> input = TOOLS::createInputs(shuffledItems[i], vocabulary);

    //Get value of shuffled item from map and convert to integer
    //0->false / 1->true
    std::map<std::string, std::string>::const_iterator pair = items.find(shuffledItems[i]);

    int target = 0;
    if((pair->second) == "True")
      target = 1;

    std::vector<double> out = rnn->forward(input);

    //Calculate loss and accuracy
    loss -= log(out[target]);
    int index = std::distance(out.begin(), std::max_element(out.begin(), out.end()));
    if(index == target)
      accuracy += 1;

    if(backprop)
    {
      out[target] -= 1;
      rnn->backprop(out);
    }
  }
  loss /= items.size();
  accuracy /= items.size();

}

int main()
{
  READER::Reader * reader = new READER::Reader();

  std::vector<std::string> trainWords = reader->read("train_data.dat");
  std::map<std::string, std::string> trainItems = reader->getItems();

  std::vector<std::string> testWords = reader->read("test_data.dat");
  std::map<std::string, std::string> testItems = reader->getItems();

  std::cout << "<----------Vocabulary size(number of words) is " <<  trainWords.size() << "---------->" << '\n';

  RNN::Rnn * rnn = new RNN::Rnn(trainWords.size(), 2, 64);

  double loss = 0.0;
  double accuracy = 0.0;

  for (int i = 0; i < 1000; i++) {
    process(trainItems, trainWords, rnn, loss, accuracy);
    if (i % 100 == 99) {
      std::cout << "Epoch " << i+1 << '\n';
      std::cout << "Train loss: " << loss << '\t' << "Train accuracy: " << accuracy << '\n';
      process(testItems, trainWords, rnn, loss, accuracy, false);
      std::cout << "Test  loss: " << loss << '\t' << "Test  accuracy: " << accuracy << '\n';
    }
  }

  delete reader;
  delete rnn;

  return 0;
}
