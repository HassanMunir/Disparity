#include "Timer.h"


double _start;
double _elapsed;
bool _running;
std::string _name;

Timer::Timer(const std::string& name)
{
    _name = name;
    _running = true;
    _start = cv::getTickCount();
}

double Timer::stop()
{
    _running = false;
    _elapsed = ((cv::getTickCount() - _start) * 1000) / cv::getTickFrequency();
    return _elapsed;
    
}

void Timer::print()
{
    if (_running)
        stop();
    std::cout << _name << ": " << _elapsed << " ms" << std::endl;
}

Timer::~Timer()
{
    
}
