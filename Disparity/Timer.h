#pragma once

#include <iostream>
#include "opencv2\contrib\contrib.hpp"


class Timer
{
public:
    Timer(const  std::string& name);
    double stop();
    void print();
    ~Timer();
};

