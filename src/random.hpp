#pragma once

#include <random>

std::default_random_engine& global_urng();

void randomize();

int pick(int from, int thru);

double pick(double from = 0, double upto = 1);
