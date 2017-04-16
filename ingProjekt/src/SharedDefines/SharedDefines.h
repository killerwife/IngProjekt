#pragma once
#include <chrono>
#include <iostream>
#include <functional>

void outputTimer(std::function<void(void)> func)
{
    auto start = std::chrono::high_resolution_clock::now();

    func();

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() << "ns\n";

}