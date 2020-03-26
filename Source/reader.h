#ifndef __READER_H__
#define __READER_H__

#include <vector>
#include <map>

namespace READER
{
  class Reader
  {
  public:
    Reader();
    ~Reader();

    //Function for reading file(returns keys from file)
    //Returned words(vector of words) are without repetitions
    std::vector<std::string> read(const std::string path);

    //Function returns items(key-value pairs)
    std::map<std::string, std::string> getItems();

  private:
    //Map for storing items(key-value pairs)
    std::map<std::string, std::string> m_items;
    //Function to split words
    void splitWords(const std::string& str, std::vector<std::string>& keys);
  };
}

#endif
