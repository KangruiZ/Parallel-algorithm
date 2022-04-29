#include <chrono>
#include <cstring>
#include <iostream>

struct Timer
{
    using clock_t = std::chrono::high_resolution_clock;
    using time_point_t = std::chrono::time_point<clock_t>;
    using elapsed_time_t = std::chrono::duration<double, std::milli>;

    time_point_t mStartTime;
    time_point_t mStopTime;
    elapsed_time_t mElapsedTime;

    void Start()
    {
        mElapsedTime = elapsed_time_t::zero();
        mStartTime = clock_t::now();
    }

    void Stop(const std::string& msg = "time:")
    {
        mStopTime = clock_t::now();
        std::chrono::duration<double, std::milli> elapsedTime = mStopTime - mStartTime;
        std::cout << "[" << msg << elapsedTime.count() << "ms]" << std::endl;
    }

};