#ifndef ___TICTOC_H___
#define ___TICTOC_H___



#include <stack>
#include <ctime>



class tictoc
{
    
public:
    
    void tic();
    void toc();
    
private:
    
    std::stack<clock_t> mTictocStack;
    
};



#endif // ___TICTOC_H___
