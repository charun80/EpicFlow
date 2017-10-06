#include "tictoc.h"
#include <iostream>



void tictoc::tic()
{
    mTictocStack.push(clock());
}



void tictoc::toc()
{
    std::cout << "Time elapsed: "
              << ((double)(clock() - mTictocStack.top())) // CLOCKS_PER_SEC
              << std::endl;
    mTictocStack.pop();
}


