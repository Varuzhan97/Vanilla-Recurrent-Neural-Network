#include <iostream>
#include <fstream> //For file reading
#include <algorithm> //For erase()
#include <cassert> //For assert()
#include <sstream> //For istringstream

#include "reader.h"

READER::Reader::Reader()
{
}

READER::Reader::~Reader()
{
}

std::vector<std::string> READER::Reader::read(const std::string path)
{
  if(!m_items.empty())
    m_items.clear();
  //Vector for words(keys)
  std::vector<std::string> keysVector;
  //Open file
  std::ifstream file(path);
  if(!file) {
    std::cout << "Cannot open input file.\n";
    exit(0);
  }
  //Read file line by line
  std::string line;
  while (std::getline(file, line)) {
    //Split string
    int pos = line.find(":");
    assert(pos != std::string::npos);
    //Get key
    std::string key = line.substr(0, pos - 1);
    //Split key by spaces
    splitWords(key, keysVector);
    //Get value
    std::string value = line.substr(pos + 1);
    //Remove spaces from value
    value.erase(remove(value.begin(), value.end(), ' '), value.end());
    m_items.insert(std::pair<std::string, std::string>(key, value));
  }
  file.close();
  return keysVector;
}

void READER::Reader::splitWords(const std::string& str, std::vector<std::string>& keys)
{
    //Used to split string around spaces
    std::istringstream ss(str);
    //Traverse through all words
    do {
        //Read a word
        std::string word;
        ss >> word;
        //Skip if word is empty
        if(word == "")
          continue;
        //Store words into vector(m_keys)(if the word is not exists)
        if (std::find(keys.begin(), keys.end(), word) == keys.end())
          keys.push_back(word);
    } while (ss);
}

std::map<std::string, std::string> READER::Reader::getItems()
{
  return m_items;
}
