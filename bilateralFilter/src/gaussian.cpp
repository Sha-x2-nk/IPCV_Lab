#include "../include/gaussian.hpp"

#include <iostream>
#include <cmath>

# define M_PI 3.14159265358979323846 /* pi */
double gaussian(double x, double sigma){
    return exp(-(x * x) / (2 * sigma * sigma)) / (sqrt(2 * M_PI) * sigma);
}