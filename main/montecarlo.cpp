#include <vector>
#include <iostream>
#include <sstream> // std::stringstream
#include <fstream> // std::fstream
#include <iomanip>
#include <limits>
#include <ctime>
#include <cmath>
#include <cfloat>
#include <random>
#include <functional>
#include <chrono>

std::vector<double> MonteCarlo(int N, int iter, double step, std::function<double(double, int)> f)
{
    // int iter = 10; // número de termos da série de fourier

    // int N = 100000;     // numero de pontos a gerar
    // double step = 1E-3; // step do montecarlo

    double theta = 0;
    double phi = 0;

    std::vector<double> Integ;
    Integ.resize(iter);
    std::fill(Integ.begin(), Integ.end(), 0);

    double Integral = 0;
    double Error = 0;

    double sum = 0., negsum = 0., err = 0., var = 0.;
    double max = -DBL_MAX, min = DBL_MAX;
    double x = -2 * M_PI;

    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<double> distribution(-2 * M_PI, 2 * M_PI);

    // Get function Max
    for (int j = 0; j < iter; j++)
    {
        x = -2 * M_PI;
        Integral = 0;
        Error = 0;
        while (x <= 2 * M_PI)
        {
            if (f(x, j) > max)
                max = f(x, j);
            if (f(x, j) < min)
                min = f(x, j);
            x = x + step;
        }
        // std::cout << "MAX: " << max << " MIN: " << min << std::endl;

        // Get Area
        double area = 4 * M_PI * std::fabs(max - min);

        std::uniform_real_distribution<double> ygen(min, max);

        for (int i = 0; i < N; i++)
        {
            double x_ = distribution(generator);
            double y_ = ygen(generator);
            if (y_ > 0 && y_ < f(x_, j))
                sum = sum + 1;
            if (y_ < 0 && y_ > f(x_, j))
                negsum = negsum + 1;
        }
        Integral = (area * (double)(sum / N) - area * (double)(negsum / N));
        Error = (area / N) * sqrt((sum - negsum) * (1 - ((sum - negsum) / N)));
        // std::cout << std::setprecision(7) << "(" << Integral << ", " << Error << ")" << std::endl;

        sum = 0;
        negsum = 0;
        Integ[j] = Integral;
    }

    return Integ;
}

void Write(std::vector<double> a, std::vector<double> b, std::string name)
{
    std::vector<std::pair<double, double>> vec;
    for (int i = 0; i < a.size(); i++)
        vec.push_back(std::make_pair(a[i], b[i]));

    std::ofstream File(name);

    for (int i = 0; i < vec.size(); i++)
        File << vec[i].first << ";" << vec[i].second << std::endl;
};

void FourierCoefficients(std::function<double(double, int)> par, std::function<double(double, int)> impar, int N, int iter, double step, std::string filename)
{
    auto Bn = MonteCarlo(N, iter, step, impar);
    auto An = MonteCarlo(N, iter, step, par);

    int k = M_PI;

    std::cout << par(k, 0) << "\t" << par(-k, 0) << std::endl;
    if (par(k, 0) == par(-k, 0))
    {
        for (int i = 0; i < Bn.size(); i++)
        {
            Bn[i] = 0;
        }
    }

    if (par(k, 0) == -par(-k, 0))
    {
        for (int i = 0; i < Bn.size(); i++)
            An[i] = 0;
    }

    Write(An, Bn, filename);
};

int main()
{
    std::function<double(double, int)> th_cos = [](double x, int i)
    { return ((x * x) * cos((i * x) / 2)) / (2 * M_PI); };

    std::function<double(double, int)> th_sin = [](double x, int i)
    { return ((x * x) * sin((i / 2) * x)) / (2 * M_PI); };

    std::function<double(double, int)> ph_cos = [](double x, int i)
    { return (x * cos((i / 2)) * x) / (2 * M_PI); };
    std::function<double(double, int)> ph_sin = [](double x, int i)
    { return (x * sin((i / 2) * x)) / (2 * M_PI); };

    FourierCoefficients(th_cos, th_sin, 1000000, 10, 1E-3, "theta_param.txt");

    return 0;
}